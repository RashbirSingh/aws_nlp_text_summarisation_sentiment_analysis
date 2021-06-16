FROM python:3.7-buster
MAINTAINER rashbirkohli@gmail.com

RUN apt-get update

RUN mkdir /var/assignment3

COPY . /var/assignment3

WORKDIR /var/assignment3

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

RUN python -m textblob.download_corpora

RUN pip install scikit-learn

RUN apt update

#WORKDIR .

#RUN python setup.py install

WORKDIR /var/assignment3

EXPOSE 80

CMD python main.py
