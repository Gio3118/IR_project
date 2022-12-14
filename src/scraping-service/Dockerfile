FROM python
ENV PYTHONBUFFERED 1

WORKDIR /scraping-service
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
