FROM coady/pylucene
ENV PYTHONBUFFERED 1

WORKDIR /search-service
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
