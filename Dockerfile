FROM nvcr.io/nvidia/deepstream-l4t

RUN apt-get update && apt-get install -y \
        pkg-config \
        zlib1g-dev \
        libwebp-dev \
        libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        cmake \
        python3.6 python3.6-dev python3-pip \
        libssl-dev \
        libffi-dev

RUN pip3 install --upgrade pipnvcr.io/nvidia/deepstream-l4t

RUN mkdir src
COPY requirements/ /app/

WORKDIR /app

RUN git clon https://github.com/davisking/dlib.git \
	cd dlib \
	mkdir build \
	cd build \
	cmake .. -DDLIB_USE_CUDA = 1 -DUSE_AVX_INSTRUCTIONS = 1 \
	cmake --build. \
	cd .. \
	python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA

RUN pip3 install -r requirements.txt
