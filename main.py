import pickle
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

count = 0

app = FastAPI(title='Credit_card_default_prediction')


# Load dv and sc at startup
with open ('credit_card_client_dataset.pkl', 'rb') as f:
    dv = pickle.load(f)
    model = pickle.load(f)
    sc = pickle.load(f)


num_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

class CreditCardClientData(BaseModel):
    LIMIT_BAL: float        
    AGE: int 
    PAY_0: int 
    PAY_2: int 
    PAY_3: int 
    PAY_4: int 
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float 
    BILL_AMT3: float
    BILL_AMT4: float            
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@app.post('/predict')
async def predict(data: CreditCardClientData):
    import xgboost as xgb # type: ignore
    global count
    count += 1
    print(f"Request {count}: {data}")

    try:
        data_dict = data.dict()
        num_data = {k: data_dict[k] for k in num_cols}
        scaled_num = sc.transform(pd.DataFrame([num_data]))
        for i, k in enumerate(num_cols):
            data_dict[k] = scaled_num[0][i]

        print(f"data_dict: {data_dict}")
        X = dv.transform([data_dict])
        feature_names=dv.get_feature_names_out().tolist()   
        dmat = xgb.DMatrix(X, feature_names=feature_names) 
        y_pred = float(model.predict(dmat)[0])
        return {"prediction": int(y_pred), "count":count}
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info", reload=True)    

