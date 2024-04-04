
ARG PYTHON_VERSION=3.10.12
FROM python:${PYTHON_VERSION}-slim 

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD exec gunicorn --bind :8080 --workers 1 --timeout 0 --threads 8 app:app
