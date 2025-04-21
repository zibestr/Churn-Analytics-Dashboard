FROM python:3.12-slim

RUN pip install --upgrade pip wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY main.py .
COPY models ./models
COPY src ./src

CMD ["python3", "main.py"]