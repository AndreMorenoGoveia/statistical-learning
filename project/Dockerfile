FROM python:3.14-slim


WORKDIR /usr/src/code/

COPY . /usr/src/code/
COPY requirements.txt /usr/src/code/
COPY requirements_torch.txt /usr/src/code/

RUN apt update && apt install -y zip vim tmux unzip htop

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

RUN python -m pip install -r requirements_torch.txt


ENTRYPOINT ["tail", "-f", "/dev/null"]

