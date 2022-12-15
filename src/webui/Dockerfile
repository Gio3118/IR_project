FROM python
ENV PYTHONBUFFERED 1

WORKDIR /webui
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
