FROM tensorflow/tensorflow:latest-gpu
RUN apt update && apt install -y cmake g++ wget unzip
RUN cd / && wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && unzip opencv.zip \
	&& cd opencv-4.x && mkdir -p build \ 
	&& cd build && cmake ..\
	&& make -j 8 && make install
RUN apt-get install -y build-essential git cmake cmake-curses-gui swig
RUN git clone https://github.com/robotology/ycm.git
RUN cd / && mkdir robotology && mv ycm/ robotology/ \
	    && cd robotology/ycm && mkdir build \
	    && cd build && cmake .. && make && make install
RUN apt-get install -y \
    libeigen3-dev \
    libace-dev \
    libedit-dev \
    libsqlite3-dev \
    libtinyxml-dev \
    qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev \
    qml-module-qtquick2 qml-module-qtquick-window2 \
    qml-module-qtmultimedia qml-module-qtquick-dialogs \
    qml-module-qtquick-controls qml-module-qt-labs-folderlistmodel \
    qml-module-qt-labs-settings \
    libqcustomplot-dev \
    libgraphviz-dev \
    libjpeg-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

RUN cd /robotology && git clone https://github.com/robotology/yarp.git \
	&& cd yarp && mkdir build && cd build \ 
    && cmake -D ENABLE_yarpcar_mjpeg=ON -D YARP_COMPILE_BINDINGS=ON -D CREATE_PYTHON=ON .. \
	&& make -j8 && make install && ldconfig
RUN cd /robotology && git clone https://github.com/robotology/icub-main.git \
	&& cd icub-main && mkdir build && cd build && cmake .. && make && make install 
RUN cd /robotology && git clone https://github.com/robotology/icub-contrib-common.git \
	&& cd icub-contrib-common && mkdir build && cd build && cmake .. && make && make install
RUN apt-get -y  install swig

#Code specific requirements
RUN pip3 install opencv-python tqdm matplotlib
RUN cd /robotology && git clone -b feat/python_module https://github.com/fbrand-new/E2Pose.git
RUN cd /robotology && git clone -b feature/yarpE2Pose https://github.com/fbrand-new/human-sensing.git
#Download the pretrained models
RUN cd /robotology/human-sensing/yarpE2Pose/pretrains && ./download.sh

#Fix gpu driver version
RUN ln -f -s libcuda.so.515.65.01 libcuda.so.1
