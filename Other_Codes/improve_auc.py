import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Optional: LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# XGBoost
import xgboost as xgb

RANDOM_STATE = 42


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop ID if present
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Ratios: BILL/Limit and PAY/BILL
    for k in range(1, 7):
        bill_col = f'BILL_AMT{k}'
        pay_col = f'PAY_AMT{k}'
        # BILL ratio to limit
        df[f'bill_ratio_{k}'] = df[bill_col] / (df['LIMIT_BAL'] + 1.0)
        # PAY ratio to bill (clip to handle outliers)
        df[f'pay_ratio_{k}'] = df[pay_col] / (df[bill_col] + 1.0)
        df[f'pay_ratio_{k}'] = df[f'pay_ratio_{k}'].clip(0, 1.5)

    # Deltas month-to-month for bills and payments
    for k in range(2, 7):
        df[f'bill_diff_{k}'] = df[f'BILL_AMT{k}'] - df[f'BILL_AMT{k-1}']
        df[f'pay_diff_{k}'] = df[f'PAY_AMT{k}'] - df[f'PAY_AMT{k-1}']

    # Aggregates
    bill_cols = [f'BILL_AMT{k}' for k in range(1, 7)]
    pay_cols = [f'PAY_AMT{k}' for k in range(1, 7)]
    df['bill_mean'] = df[bill_cols].mean(axis=1)
    df['bill_std'] = df[bill_cols].std(axis=1)
    df['pay_mean'] = df[pay_cols].mean(axis=1)
    df['pay_std'] = df[pay_cols].std(axis=1)

    # Current utilization proxy
    df['current_util'] = df['BILL_AMT6'] / (df['LIMIT_BAL'] + 1.0)

    # Ensure PAY_* are integers (if not already)
    for k in [0, 2, 3, 4, 5, 6]:
        col = f'PAY_{k}'
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def split_data(df: pd.DataFrame):
    y = df['default'].astype(int)
    X = df.drop(columns=['default'])

    # Global split 80/20
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # From full train, split 75/25 -> total 60/20/20
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_full_train
    )

    features = list(X_train.columns)
    return X_train, y_train, X_val, y_val, X_test, y_test, features


def train_xgboost(X_train, y_train, X_val, y_val, features):
    # scale_pos_weight = negative / positive (computed on training fold)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 5,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'verbosity': 1,
        'seed': RANDOM_STATE,
    }

    watchlist = [(dtrain, 'train'), (dval, 'valid')]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    return model


def evaluate_xgboost(model, X_val, y_val, X_test, y_test, features):
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    if hasattr(model, 'best_iteration') and model.best_iteration is not None:
        y_val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
        y_test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
    else:
        y_val_pred = model.predict(dval)
        y_test_pred = model.predict(dtest)

    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    return val_auc, test_auc, y_val_pred, y_test_pred


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_rf(model, X_val, y_val, X_test, y_test):
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    return val_auc, test_auc, y_val_pred, y_test_pred


def train_lightgbm(X_train, y_train, X_val, y_val):
    if not HAS_LGBM:
        return None
    model = LGBMClassifier(
        objective='binary',
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_lambda=1.0,
        n_estimators=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        # Handle imbalance
        class_weight='balanced',
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        verbose=100,
        callbacks=None,
    )
    return model


def evaluate_lightgbm(model, X_val, y_val, X_test, y_test):
    if model is None:
        return None
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    return val_auc, test_auc, y_val_pred, y_test_pred


def main():
    csv_path = os.path.join(os.getcwd(), 'UCI_Credit_Card.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found at {csv_path}')

    print('Loading data...')
    df = load_data(csv_path)

    print('Engineering features...')
    df_fe = feature_engineering(df)

    print('Splitting data...')
    X_train, y_train, X_val, y_val, X_test, y_test, features = split_data(df_fe)

    print(f'Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}')

    # XGBoost
    print('\nTraining XGBoost with early stopping...')
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, features)
    xgb_val_auc, xgb_test_auc, _, _ = evaluate_xgboost(xgb_model, X_val, y_val, X_test, y_test, features)
    print(f'XGBoost AUC - Val: {xgb_val_auc:.4f} | Test: {xgb_test_auc:.4f}')

    # Feature importance (top 20)
    try:
        booster = xgb_model
        fmap = booster.get_score(importance_type='gain')
        imp_df = pd.DataFrame({'feature': list(fmap.keys()), 'gain': list(fmap.values())})
        imp_df.sort_values('gain', ascending=False, inplace=True)
        print('\nTop 20 XGBoost features (by gain):')
        print(imp_df.head(20))
    except Exception:
        pass

    # RandomForest
    print('\nTraining RandomForest (balanced)...')
    rf_model = train_random_forest(X_train, y_train)
    rf_val_auc, rf_test_auc, _, _ = evaluate_rf(rf_model, X_val, y_val, X_test, y_test)
    print(f'RandomForest AUC - Val: {rf_val_auc:.4f} | Test: {rf_test_auc:.4f}')

    # LightGBM (optional)
    if HAS_LGBM:
        print('\nTraining LightGBM...')
        lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
        lgbm_res = evaluate_lightgbm(lgbm_model, X_val, y_val, X_test, y_test)
        if lgbm_res is not None:
            lgbm_val_auc, lgbm_test_auc, _, _ = lgbm_res
            print(f'LightGBM AUC - Val: {lgbm_val_auc:.4f} | Test: {lgbm_test_auc:.4f}')
    else:
        print('\nLightGBM not available; skipping.')

    print('\nDone.')


if __name__ == '__main__':
    main()
