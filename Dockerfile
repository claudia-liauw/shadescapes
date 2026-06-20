# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.10.14
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Set environment variables to optimize uv in Docker
# Compile bytecode for faster startup times
ENV UV_COMPILE_BYTECODE=1 
# Prevent uv from complaining about virtual environment activation
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"
# Put the virtual environment in the PATH so we can run Python directly
ENV PATH="/app/.venv/bin:$PATH"

# 1. Copy ONLY dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# 2. Install dependencies (without application code)
# --frozen ensures we use exact versions from uv.lock
# --no-dev excludes testing/development packages
# --no-install-project prevents uv from trying to install the app code itself right now
RUN uv sync --frozen --no-dev --no-install-project

# 3. Copy the rest of the application code
COPY . .

# 4. Install the project itself (if it's a package)
RUN uv sync --frozen --no-dev

# Expose your app's port (e.g., FastAPI/Django)
EXPOSE 8000

# Run the application.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
