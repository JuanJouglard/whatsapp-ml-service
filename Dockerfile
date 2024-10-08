FROM python:3.10-slim

WORKDIR /ml

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
