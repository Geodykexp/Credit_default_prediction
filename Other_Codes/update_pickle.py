import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('UCI_Credit_Card.csv')
num_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
sc = StandardScaler()
df[num_cols] = sc.fit_transform(pd.DataFrame(df[num_cols]))

# Load existing dv and model
with open('credit_card_client_dataset.pkl', 'rb') as f:
    dv = pickle.load(f)
    model = pickle.load(f)
    _ = pickle.load(f)  # old sc

# Save with new sc
with open('credit_card_client_dataset.pkl', 'wb') as f:
    pickle.dump(dv, f)
    pickle.dump(model, f)
    pickle.dump(sc, f)

print('Pickle updated with correct sc')