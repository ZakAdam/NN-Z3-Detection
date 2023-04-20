# syntax=docker/dockerfile:1
FROM dockerhub.deepmagic.sk/stu/dev-base:latest

# Instalujeme inno packages
RUN apt-get update --fix-missing
RUN apt-get install apt-transport-https
RUN apt-get install ca-certificates
RUN update-ca-certificates

# User created in parent container
ARG USERNAME=user

# ADD YOUR ID HERE - id into console
ARG USER_UID=XXXX
ARG USER_GID=$USER_UID

WORKDIR /app
COPY requirements.txt requirements.txt

# Nainstalujeme requirementy
RUN pip install -r requirements.txt
USER user

