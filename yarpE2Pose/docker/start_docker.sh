TF_IMG=fbrand-new/e2pose:devel

docker run --rm --gpus=all --network host --pid host -e TF_FORCE_GPU_ALLOW_GROWTH=true\
  -v /etc/localtime:/etc/localtime -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY=$DISPLAY\
  -v /etc/hosts:/etc/hosts -e QT_X11_NO_MITSHM=1\
  --privileged\
  -e PYTHONPATH="/robotology/yarp/build/lib/python3:/robotology/E2Pose" \
  -it $TF_IMG bash
