import pickle
import xgboost as xgb
import pandas as pd

with open ('credit_card_client_dataset.pkl', 'rb') as f:
    dv = pickle.load(f)
    model = pickle.load(f) 
    sc = pickle.load(f)

num_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

client = {
    "LIMIT_BAL": 20000.0,    
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913.0,
    "BILL_AMT2": 3102.0,
    "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0,
    "BILL_AMT5": 0.0,
    "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 689.0,
    "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0,
    "PAY_AMT5": 0.0,
    "PAY_AMT6": 0.0
}

print("Client data:", client)

# Scale
num_data = {k: client[k] for k in num_cols}
print("Num data:", num_data)
scaled_num = sc.transform(pd.DataFrame([num_data]))
print("Scaled num:", scaled_num)
for i, k in enumerate(num_cols):
    client[k] = scaled_num[0][i]

print("Scaled client:", client)

X = dv.transform([client])
print("X shape:", X.shape)
feature_names = dv.get_feature_names_out().tolist()
print("Feature names:", feature_names)
dmat = xgb.DMatrix(X, feature_names=feature_names)
y_pred = float(model.predict(dmat)[0])
print("Prediction:", y_pred)