FROM python:3.10
ENV PYTHONBUFFERED 1

WORKDIR /sentiment-analysis-service
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
