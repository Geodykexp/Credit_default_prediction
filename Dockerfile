# DOCKERFILE FOR LOCAL TESTING WITH UVIORN AND UV

# FROM python:3.13-slim-bookworm

# COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# WORKDIR /app 

# # Environmental variable (Experimental)
# ENV PATH="/app/ .venv/bin:$PATH"

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

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

COPY "pyproject.toml" "uv.lock" ./

RUN uv sync --locked

COPY "Lambda_function.py" "credit_card_client_dataset.pkl" ./

CMD ["Lambda_function.lambda_handler"]

# CMD ["uv", "run", "--", "Lambda_function.lambda_handler"]
