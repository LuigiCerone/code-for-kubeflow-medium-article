# Generated by KFP.

FROM python:3.8

WORKDIR /usr/local/src/kfp/components
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir kfp==1.8.19
COPY . .
