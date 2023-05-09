IMG=fbrand-new/mmpose:devel

docker run -it --rm --gpus=all --network host --pid host\
  -v /etc/localtime:/etc/localtime -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY=$DISPLAY\
  -e QT_X11_NO_MITSHM=1 -v ~/misc:/root/misc -v ~/robotology/human-sensing/yarpPose:/root/yarpPose\
  -e PYTHONPATH="/root/yarpPose" $IMG bash
