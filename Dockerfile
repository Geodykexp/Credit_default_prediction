# DOCKERFILE FOR LOCAL TESTING WITH UVICORN AND UV

# FROM python:3.12-slim-bookworm

# COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# WORKDIR /app 

# # Environmental variable (Experimental)
# ENV PATH="/app/.venv/bin:$PATH"

# # Copy project files
# COPY "pyproject.toml" "uv.lock" ./

# # Sync dependencies
# RUN uv sync --locked

# # Copy application files
# COPY "main.py" "credit_card_client_dataset.pkl" ./

# EXPOSE 8000

# # Run uvicorn with 0.0.0.0 to accept external connections
# CMD ["uv", "run", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# DOCKERFILE FOR AWS LAMBDA DEPLOYMENT

FROM public.ecr.aws/lambda/python:3.12

# Install uv
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# Set working directory to Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy project files
COPY "pyproject.toml" "uv.lock" ./

# Install dependencies into the system site-packages for Lambda
RUN uv pip install --system --requirement pyproject.toml

# Copy application files
COPY "Lambda_function.py" "credit_card_client_dataset.pkl" ./

# Set the CMD to your handler
CMD ["Lambda_function.lambda_handler"]
