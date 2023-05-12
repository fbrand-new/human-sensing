ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

ARG CMAKE_GENERATOR="Unix Makefiles"
ARG BUILD_TYPE="Release"
ARG CMAKE_EXTRA_OPTIONS=-j8
ARG INSTALL_DIR="/usr/local"
ARG SUPERBUILD_RELEASE="releases/2023.02"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# FROM pytorch/pytorch:latest

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y wget git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx vim\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMEngine and MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0.0"

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN git checkout main
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN pip install mmdet
RUN pip install opencv-python

# Install yarp
RUN cd / && git clone https://github.com/robotology/robotology-superbuild &&\
    cd robotology-superbuild &&\
    git checkout ${SUPERBUILD_RELEASE} &&\
    apt update && ./scripts/install_apt_dependencies.sh &&\
    mkdir build && cd build &&\
    cmake ..  \ 
        -G ${CMAKE_GENERATOR} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DYCM_EP_INSTALL_DIR=${INSTALL_DIR} &&\
    cmake --build . -- ${CMAKE_EXTRA_OPTIONS}

RUN cd ${PROJECTS_DIR}/robotology-superbuild/src/YCM && \
    git checkout master && \
    git pull --rebase

RUN cd ${PROJECTS_DIR}/robotology-superbuild/src/YARP && \
    git checkout master && \
    git pull --rebase && \
    cmake . \                                                                                                                                              1         -G "$CMAKE_GENERATOR" \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_yarpcar_mjpeg=ON \
    -DENABLE_yarpmod_RGBDSensorClient=ON \
    -DENABLE_yarppm_bottle_compression_zlib=ON \
    -DENABLE_yarppm_image_compression_ffmpeg=ON \
    -DENABLE_YARP_COMPILE_BINDINGS=ON \
    -DCREATE_PYTHON=ON

COPY . /yarpPose
RUN /yarpPose/download_models.sh

COPY docker/entrypoint.sh /
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"] 
