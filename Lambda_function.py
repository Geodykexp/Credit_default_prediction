import pickle
import json
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load model at cold start
with open('credit_card_client_dataset.pkl', 'rb') as f:
    dv = pickle.load(f)
    model = pickle.load(f)
    sc = pickle.load(f)

num_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

def lambda_handler(event, context):
    try:
        data = json.loads(event['body'])  # Assuming API Gateway event
        num_data = {k: data[k] for k in num_cols}
        scaled_num = sc.transform(pd.DataFrame([num_data]))
        for i, k in enumerate(num_cols):
            data[k] = scaled_num[0][i]
        X = dv.transform([data])
        dmat = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
        y_pred = int(model.predict(dmat)[0])
        return {'statusCode': 200, 'body': json.dumps({'prediction': y_pred})}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}