# AWS Deployment Guide: Credit Default Prediction

This guide provides step-by-step instructions to deploy the Credit Card Default Prediction model as a serverless API on AWS using Lambda and ECR.

## Prerequisites
- [AWS CLI](https://aws.amazon.com/cli/) installed and configured (`aws configure`).
- [Docker](https://www.docker.com/) installed and running.
- An AWS Account with permissions for ECR, Lambda, and API Gateway.

---

## Step 1: Create an ECR Repository
Create a repository in Amazon Elastic Container Registry (ECR) to store your Docker image.

### Option A: Using AWS Management Console (GUI)
1.  Open the [Amazon ECR Console](https://console.aws.amazon.com/ecr/repositories).
2.  Click **Create repository**.
3.  **Visibility settings**: Choose **Private**.
4.  **Repository name**: `credit-default-prediction`.
5.  Click **Create repository**.

### Option B: Using AWS CLI
```bash
# Variables
REGION="us-east-1"
REPO_NAME="credit-default-prediction"

# Create the repository
aws ecr create-repository --repository-name $REPO_NAME --region $REGION
```

Take note of the `repositoryUri` from the output (e.g., `<aws_account_id>.dkr.ecr.<region>.amazonaws.com/credit-default-prediction`).

---

## Step 2: Build and Tag the Docker Image
Build the image specifically for the `linux/amd64` architecture to ensure compatibility with AWS Lambda.

```bash
# Variables
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"

# Build the image
docker build --platform linux/amd64 -t $REPO_NAME .

# Tag the image for ECR
docker tag $REPO_NAME:latest $ECR_URI:latest
```

---

## Step 3: Push the Image to ECR
Authenticate Docker with AWS and push the image.

```bash
# Authenticate
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Push
docker push $ECR_URI:latest
```

---

## Step 4: Create the Lambda Function
Create a Lambda function using the image you just pushed.

1.  Go to the [AWS Lambda Console](https://console.aws.amazon.com/lambda/).
2.  Click **Create function**.
3.  Select **Container image**.
4.  **Function name**: `predict-credit-default`.
5.  **Container image URI**: Browse for the image you pushed in Step 3.
6.  **Architecture**: Ensure `x86_64` is selected (matches our `--platform linux/amd64` build).
7.  **Execution role**: Choose "Create a new role with basic Lambda permissions".
8.  Click **Create function**.

> [!TIP]
> **Timeout & Memory**: Under the "Configuration" tab -> "General configuration", increase the **Timeout** to `30 seconds` and **Memory** to `512 MB` or `1024 MB` to handle model loading (XGBoost) efficiently.

---

## Step 5: Expose via API Gateway
To make the Lambda reachable via HTTPS, create a REST API.

1.  Go to the [API Gateway Console](https://console.aws.amazon.com/apigateway/).
2.  Click **Create API** -> **HTTP API**.
3.  Click **Add integration** and select **Lambda**.
4.  Select your Lambda function (`predict-credit-default`).
5.  **API Name**: `CreditDefaultAPI`.
6.  Click **Next**, then **Next** (keep defaults for routes like `$default`), then **Create**.
7.  Copy the **Invoke URL** from the dashboard.

---

## Step 6: Test the Deployment
Using `curl` or a tool like Postman, send a test request to your new endpoint.

```bash
# Replace <INVOKE_URL> with your actual URL
curl -X POST "<INVOKE_URL>/predict" \
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

---

## Troubleshooting
- **Error: Image is not supported**: Ensure you used `--platform linux/amd64` when building and selected `x86_64` in Lambda.
- **ModuleNotFoundError**: Ensure the `Dockerfile` uses `uv pip install --system`.
- **Execution Role**: Ensure the Lambda's IAM role has permissions to execute.
