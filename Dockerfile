FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel


RUN echo export PATH="$HOME/.local/bin:$PATH"

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get -y install wget
RUN apt-get install -y git build-essential \
  gcc make yasm autoconf curl \
  automake cmake libtool \
  checkinstall libmp3lame-dev \
  pkg-config libunwind-dev \
  zlib1g-dev libssl-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install git+https://github.com/S-aiueo32/lpips-pytorch.git

RUN pip install --upgrade pip
ADD requirements.txt /app/
RUN pip install -r requirements.txt
# COPY . /app

RUN export LC_ALL="en_US.UTF-8"
RUN export LC_CTYPE="en_US.UTF-8"

ENTRYPOINT ["tail", "-f", "/dev/null"]

CMD ["start"]