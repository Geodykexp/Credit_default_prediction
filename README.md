# Credit Card Default Prediction

This project predicts whether a client will default on their next payment using the UCI Credit Card dataset. The project includes a FastAPI service and AWS Lambda function for inference. The repository also contains notebooks and scripts used during model development.

The repository includes:

- An online inference API with FastAPI (local or containerized)
- A serverless inference handler for AWS Lambda
- A pre-trained model bundle (DictVectorizer, XGBoost model, and StandardScaler) serialized in a single pickle file
- Notebooks and scripts used during model development


## Table of Contents
- Features
- Project Structure
- Model Overview
- Requirements
- Quickstart (Local)
- API Reference
- Docker
- AWS Lambda Deployment
- Testing
- Data Schema
- Development Notes
- Troubleshooting
- License


## Features
- FastAPI service with a single /predict endpoint
- Input validation with Pydantic
- Numeric feature scaling using StandardScaler
- Vectorization with DictVectorizer and prediction using XGBoost
- Consistent preprocessing between local API and Lambda
- Containerized workflows via Docker


## Project Structure
```
credit_card_client_dataset_prediction/
├─ Dockerfile                      # Build for local API (commented) or AWS Lambda image
├─ Lambda_function.py              # AWS Lambda handler
├─ main.py                         # FastAPI application
├─ pyproject.toml                  # Project metadata and Python dependencies
├─ uv.lock                         # Locked dependency graph for uv
├─ credit_card_client_dataset.pkl  # Serialized DictVectorizer, XGBoost model, StandardScaler (in this order)
├─ credit_card_client_dataset.py   # End-to-end training script that produces the pkl
├─ credit_card_client_dataset.ipynb# Notebook with EDA/setup mirroring the script
├─ UCI_Credit_Card.csv             # Dataset (raw)
├─ test_predict.py                 # Simple request example/test
├─ test_server.py                  # Server tests/validation helpers
├─ validate_server.py              # Local validation helper
└─ codes/                          # Additional notebooks and utilities
```


## Model Overview
- Dataset: UCI Credit Card Default (Taiwan)
- Objective: Predict whether a client will default on their next payment
- Core algorithm: XGBoost binary classifier trained on scaled numeric features
- Preprocessing:
  - Numeric columns are standardized with sklearn's StandardScaler
  - Feature transformation via sklearn's DictVectorizer
- Artifacts:
  - DictVectorizer (feature transformation)
  - StandardScaler (numeric scaling)
  - XGBoost model
- Packaging: All three artifacts are serialized together in credit_card_client_dataset.pkl in the following order: DictVectorizer, model, scaler.

Training summary (from credit_card_client_dataset.py):
- Data load: UCI_Credit_Card.csv
- Target: default
- Split: 60/20/20 into train/val/test via two-stage train_test_split with fixed seeds
- Scaling: StandardScaler fit on training data numeric columns (num_cols auto-selected by nunique > 7)
- Baselines: SVM, AdaBoost, GradientBoosting, LightGBM, LogisticRegression, DecisionTree, RandomForest evaluated (accuracy, classification report, confusion matrix, ROC-AUC)
- Final model: XGBoost trained with early stopping and eval_metric=auc; subsample=0.8, colsample_bytree=0.8; learning rate eta=0.1; seed/random_state fixed for reproducibility
- Persistence: dv, model, sc saved into credit_card_client_dataset.pkl; validated by reload checks

Note: The training code and experiments are contained in the notebooks and the credit_card_client_dataset.py script within the repository.


## Requirements
- Python: 3.13+ locally (per pyproject.toml)
- Dependencies (managed via uv/pyproject):
  - fastapi
  - uvicorn
  - pandas
  - scikit-learn
  - xgboost (used at runtime by main.py and Lambda_function.py)
  - seaborn, matplotlib (used in training/EDA)
  - lightgbm (optional baseline in training script)

If using Docker or AWS Lambda image, dependencies are resolved inside the image via uv.


## Quickstart (Local)
There are two main ways to run locally.

1) Using your local Python environment
- Ensure Python 3.13+ is installed.
- Install dependencies:
  - With uv (recommended):
    - Install uv: https://docs.astral.sh/uv/
    - From the project root: `uv sync`
  - Or with pip:
    - `pip install -r requirements.txt` (if you export one) or install from pyproject via `pip install .`
- Start the server:
  - `uvicorn main:app --host 0.0.0.0 --port 8000`
- Open API docs: http://127.0.0.1:8000/docs

2) Using Docker (local API)
- See Docker section below. The Dockerfile includes a commented stage for local FastAPI running with Uvicorn.


## API Reference
Base URL (local): http://127.0.0.1:8000

- POST /predict
  - Description: Predicts default (0 or 1) from client attributes.
  - Request body (JSON): see Data Schema section for fields and types.
  - Response 200:
    - `{ "prediction": <0 or 1>, "count": <int> }` for FastAPI
  - Response codes:
    - 200: Success
    - 500: Server error with details

Example request:
```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
  }'
```


## Code Examples

Python client (call the FastAPI endpoint):
```
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "LIMIT_BAL": 200000,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}
resp = requests.post(url, json=payload, timeout=10)
print(resp.status_code, resp.json())
```

Minimal training and saving artifacts (excerpt from credit_card_client_dataset.py):
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

# 1) Load data and select numeric columns
DF = pd.read_csv("UCI_Credit_Card.csv")
DF = DF.drop(columns=["ID"])  # drop identifier
num_cols = [c for c in DF.columns if DF[c].nunique() > 7]

# 2) Train/val split and scaling
train_df, test_df = train_test_split(DF, test_size=0.2, random_state=1)
train_df, val_df  = train_test_split(train_df, test_size=0.25, random_state=1)

sc = StandardScaler()
train_df[num_cols] = sc.fit_transform(train_df[num_cols])
val_df[num_cols]   = sc.transform(val_df[num_cols])

y_train = train_df['default'].values
y_val   = val_df['default'].values

# 3) DictVectorizer
train_dicts = train_df[num_cols].to_dict(orient='records')
val_dicts   = val_df[num_cols].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val   = dv.transform(val_dicts)

# 4) Train XGBoost with early stopping on AUC
features = dv.get_feature_names_out().tolist()
dtrain   = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval     = xgb.DMatrix(X_val,   label=y_val,   feature_names=features)

params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 1,
}
model = xgb.train(params, dtrain, num_boost_round=200,
                  evals=[(dtrain, 'train'), (dval, 'val')],
                  early_stopping_rounds=10, verbose_eval=False)

# 5) Persist artifacts in required order
with open('credit_card_client_dataset.pkl', 'wb') as f:
    pickle.dump(dv, f)
    pickle.dump(model, f)
    pickle.dump(sc, f)
```

Load pickle and run a one-off prediction (local Python):
```
import pickle
import pandas as pd
import xgboost as xgb

# Load artifacts (order matters)
with open('credit_card_client_dataset.pkl', 'rb') as f:
    dv = pickle.load(f)
    model = pickle.load(f)
    sc = pickle.load(f)

num_cols = [
    'LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
]

sample = {
    "LIMIT_BAL": 200000, "AGE": 35, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
    "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 3913, "BILL_AMT2": 3102,
    "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0,
    "PAY_AMT5": 0, "PAY_AMT6": 0
}

scaled_num = sc.transform(pd.DataFrame([sample[k] for k in num_cols]).T)
for i, k in enumerate(num_cols):
    sample[k] = scaled_num[0][i]

X = dv.transform([sample])
dmat = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
y_pred = int(model.predict(dmat)[0])
print({'prediction': y_pred})
```

Local test of the AWS Lambda handler:
```
import json
from Lambda_function import lambda_handler

payload = {
    "LIMIT_BAL": 200000,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}

# API Gateway proxy event structure
event = {"body": json.dumps(payload)}
result = lambda_handler(event, None)
print(result)
```

## Docker
The Dockerfile supports two workflows:

- Local FastAPI (commented section at top):
  - Base: python:3.13-slim-bookworm
  - Installs dependencies via uv
  - Runs `uvicorn main:app` on port 8000

- AWS Lambda (active section):
  - Base: public.ecr.aws/lambda/python:3.12
  - Installs dependencies via uv using pyproject.toml and uv.lock
  - Copies Lambda_function.py and credit_card_client_dataset.pkl
  - Entry point: Lambda_function.lambda_handler

To run a local API image (uncomment the first stage in Dockerfile):
- Build: `docker build -t cc-default-api:local .`
- Run: `docker run --rm -p 8000:8000 cc-default-api:local`
- Open: http://127.0.0.1:8000/docs

To build the Lambda image (as currently configured):
- `docker build -t cc-default-lambda:latest .`
- Test locally with lambda runtime emulator (optional) or deploy to ECR/Lambda.


## AWS Lambda Deployment
1. Build a Lambda-compatible image for the correct architecture:
   - `docker build --platform linux/amd64 -t <your-ecr-repo>:<tag> .`
2. Authenticate to ECR and push image.
3. Create or update a Lambda function with the image.
4. Set the handler to `Lambda_function.lambda_handler` (already set by CMD).
5. Expose via API Gateway to get an HTTPS endpoint.

Handler contract (API Gateway proxy integration):
- event["body"] must be a JSON string with the same schema as the FastAPI request body.
- Response: `{ "prediction": 0|1 }` when statusCode == 200.


## Testing
- With the FastAPI server running locally:
  - Use the Swagger UI: http://127.0.0.1:8000/docs
  - Or send the example curl above.
- Repo includes helper scripts/tests:
  - test_predict.py, test_server.py, validate_server.py (adjust endpoints/ports if needed).


## Data Schema
The API expects a JSON object with the following fields. Types reflect Pydantic model in main.py:
- LIMIT_BAL: float
- AGE: int
- PAY_0: int
- PAY_2: int
- PAY_3: int
- PAY_4: int
- PAY_5: int
- PAY_6: int
- BILL_AMT1: float
- BILL_AMT2: float
- BILL_AMT3: float
- BILL_AMT4: float
- BILL_AMT5: float
- BILL_AMT6: float
- PAY_AMT1: float
- PAY_AMT2: float
- PAY_AMT3: float
- PAY_AMT4: float
- PAY_AMT5: float
- PAY_AMT6: float

Notes:
- Numeric fields are scaled internally using the saved StandardScaler before vectorization.
- Any missing or extra fields will result in a validation error.


## Development Notes
- Entry points:
  - Local API: main.py
  - Lambda: Lambda_function.py
- Artifacts loading order in both entry points: DictVectorizer, model, scaler from credit_card_client_dataset.pkl
- The FastAPI service logs a request count with each call and returns it in the response for observability.
- The XGBoost dependency is imported inside the predict route in main.py to optimize cold start.

Reproducing the model (training):
- Run script: `python credit_card_client_dataset.py`
  - Loads UCI_Credit_Card.csv
  - Scales numeric features, vectorizes with DictVectorizer
  - Trains XGBoost with early stopping (monitors AUC)
  - Saves dv, model, sc to credit_card_client_dataset.pkl
- Alternatively, use the notebook credit_card_client_dataset.ipynb for step-by-step EDA and training.

About credit_card_client_dataset.pkl:
- Contains three pickled objects written in this order: DictVectorizer, XGBoost Booster, StandardScaler
- Both main.py and Lambda_function.py expect this order during load

Suggested enhancements:
- Add CI to build/test Docker images
- Add hyperparameter search and model tracking
- Add pydantic field constraints and example schema
- Provide a requirements.txt export for non-uv users


## Troubleshooting
- ImportError: xgboost not found
  - Ensure xgboost is installed in your environment or included in the container image.
- Model not found / pickle load error
  - Confirm credit_card_client_dataset.pkl exists in the project root and was copied into the image for Docker/Lambda.
- 500 errors from API
  - Inspect server logs. Often due to schema mismatches or missing fields.
- Port conflicts running locally
  - Change port with `--port` when launching uvicorn.


## License
This project is provided as-is without a specific license declared. Add a LICENSE file if you intend to distribute or open-source under specific terms.
