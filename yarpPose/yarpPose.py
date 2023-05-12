import numpy as np
import yarp

class PoseEstimation():

    def __init__():
        pass
        
    def inference():
        pass

    def configure():
        pass

        image_h = READ
        image_w = READ

        yarp.Network.init()

        in_port = yarp.BufferedPortImageRgb()
        in_port_name = %%MODULENAME%%+"/image:i"
        in_port.open(in_port_name)

        out_port = yarp.BufferedPortImageRgb()
        out_port_name = %%MODULENAME%%+"/image:o"
        out_port.open(in_port_name)


        in_buf_array = np.ones((image_h,image_w,3), dtype = np.uint8)
        in_buf_image = yarp.ImageRgb()
        in_buf_image.resize(image_w,image_h)
        in_buf_image.setExternal(in_buf_array,in_buf_array.shape[1],in_buf_array.shape[1])

        out_buf_array = np.ones((image_h,image_w,3), dtype = np.int8)
        out_buf_image = yarp.ImageRgb()
        out_buf_image.resize(image_w,image_h)
        out_buf_image.setExternal(out_buf_array,out_buf_array.shape[1],out_buf_array.shape[1])

        while True:

            received_image = in_port.read()
            in_buf_image.copy(received_image)

            assert in_buf_array.__array_interface__['data'][0] == in_buf_image.getRawImage().__int__()

            
