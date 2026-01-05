curl -X POST \
     -H "Content-Type: application/json" \
     -d '{
         "body": {
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
     }' \
     http://localhost:9000/2015-03-31/functions/function/invocations