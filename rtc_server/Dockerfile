FROM nvidia/cuda:9.2-base-ubuntu16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt-get install -y ffmpeg
RUN apt update && apt install -y ffmpeg libav-tools x264 x265

# aiortc dependencies
RUN apt install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config
RUN apt-get install -y python3-dev libxml2-dev libxslt-dev

# python and opencv stuff
RUN apt-get -y --no-install-recommends install \
python3 \
python3-pip \

libglib2.0-0 \
libxext6 \
libsm6 \
libxrender-dev \
python3-setuptools

RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip
RUN pip3 install \
wheel \
attrs \
cffi \
crc32c \
cryptography \
pyee \
pylibsrtp \
pyopenssl \
sphinxcontrib-asyncio \
opencv-python \
aiohttp \
av \
aiortc


WORKDIR "/rtc_server"
ENTRYPOINT ["bash"]
