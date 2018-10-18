FROM python:3.7

RUN apt-get update -qq \
    && apt-get upgrade -qq -y \
    && apt-get clean

RUN mkdir /code
WORKDIR /code

COPY pdmodel.py .
COPY ml2p-0.0.1.tar.gz .

RUN pip install \
    ml2p-0.0.1.tar.gz

EXPOSE 8000
ENTRYPOINT ["python3", "pdmodel.py"]
