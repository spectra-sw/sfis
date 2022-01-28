ARG VERSION=1.9.2-py38-cpu
FROM  rayproject/ray:${VERSION}

RUN sudo apt-get update && sudo apt-get install -y zlib1g-dev libgl1-mesa-dev libgtk2.0-dev python3-opencv neovim

RUN sudo apt-get clean

RUN pip3 install --no-cache -U scipy \
Pillow \
elasticsearch \
imutils \
nest_asyncio \
flask \
opencv-python

WORKDIR /home/service
COPY Serveargument.py /home/service

WORKDIR /home/webservice
COPY Webservice.py /home/webservice

COPY init.sh /home

RUN sudo chmod +x /home/init.sh

WORKDIR /home/

CMD ["bash"]
