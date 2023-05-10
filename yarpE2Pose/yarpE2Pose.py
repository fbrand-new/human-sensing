import numpy as np
import sys
import yarp
import tensorflow as tf
# from e2pose.models.model import E2PoseModel
# from e2pose.models import layers
from e2pose.utils import draw
# from e2pose.utils.define import POSE_DATASETS
import os
from pathlib import Path
import cv2

yarpE2Pose_path = Path(__file__).parent

class E2PoseInference():
    def __init__(self, graph_path):
        print(graph_path)

    def decode(self, pred, src_hw, th=0.5):
        pv, kpt      = pred  #pv: person probability, kpt: keypoints probability
        pv           = np.reshape(pv[0], [-1])
        kpt          = kpt[0][pv>=th]
        kpt[:,:,-1] *= src_hw[0] #Most likely this is just a resizing with respect to the original dimensions of the image
        kpt[:,:,-2] *= src_hw[1]
        # kpt[:,:,-3] *= 2 #TODO: I dont think this multiplication is correct. That column should hold the confidence and it is a probability measure
        ret = []
        for human in kpt:
            mask   = np.stack([(human[:,0] >= th).astype(np.float32)], axis=-1)
            human *= mask
            human  = np.stack([human[:,_ii] for _ii in [1,2,0]], axis=-1)
            ret.append({'keypoints': np.reshape(human, [-1]).tolist(), 'category_id':1})
        return ret


class E2PoseInference_by_pb(E2PoseInference):
    def __init__(self, graph_path):
        with tf.io.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name='e2pose')
        self.persistent_sess = tf.compat.v1.Session(graph=self.graph, config=None)

        input_op           = [op for op in self.graph.get_operations() if 'inputimg' in op.name][0]
        out_pv_op          = [op for op in self.graph.get_operations() if 'pv/concat' in op.name][-1]
        out_kpt_op         = [op for op in self.graph.get_operations() if 'kvxy/concat' in op.name][-1]        
        self.tensor_image  = input_op.outputs[0]
        self.tensor_output = [out_pv_op.outputs[0], out_kpt_op.outputs[0]]

    @property
    def inputs(self):
        return [self.tensor_image]

    def __call__(self, x, training=False):
        return self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: x})

class yarpE2PoseModule(yarp.RFModule):

    def configure(self, rf):
        
        self.model_name = rf.find("model").asString()
        self.dataset = rf.find("dataset").asString()
        self.image_width = rf.find("width").asInt16()
        self.image_height = rf.find("height").asInt16()
        self.face_keypoints = rf.find("face-keypoints").asInt16()
        self.period = rf.find("period").asInt16()
       
        # Get net input size from model name
        self.model_input_size = list(map(int,self.model_name.split("/")[2].split("x"))) 

        # Load the model
        self.model = E2PoseInference_by_pb(os.path.join(yarpE2Pose_path,"models",self.model_name))
        self.painter = draw.Painter(self.dataset)

        #Input port rgb image
        self.in_port_image = yarp.BufferedPortImageRgb()
        self.in_port_image.open("/e2pose/image:i")
        self.in_buf_array = np.zeros((self.image_height, self.image_width, 3), dtype = np.uint8)
        self.in_buf_image = yarp.ImageRgb()
        self.in_buf_image.resize(self.image_width,self.image_height)
        self.in_buf_image.setExternal(self.in_buf_array.data, self.in_buf_array.shape[1], self.in_buf_array.shape[0])

        #Output port rgb image
        self.out_port_image = yarp.Port()
        self.out_port_image.open('/e2pose/image:o')
        self.out_buf_array = np.ones((self.image_height, self.image_width, 3), dtype = np.uint8)
        self.out_buf_image = yarp.ImageRgb()
        self.out_buf_image.resize(self.image_width,self.image_height)
        self.out_buf_image.setExternal(self.out_buf_array.data, self.out_buf_array.shape[1], self.out_buf_array.shape[0])

        #Outport keypoints
        self.out_port_target = yarp.Port()
        self.out_port_target.open('/e2pose/target:o')

        # COCO mapping
        self.coco_keypoints = {0:"Nose",6:"RShoulder",8:"RElbow",10:"RWrist",5:"LShoulder",7:  "LElbow",9: "LWrist",12: "RHip",
                              14:"RKnee",16:"RAnkle",11:"LHip",13:"LKnee",15:"LAnkle",2: "REye",1:"LEye",4: "REar",3: "LEar"}
        
        # These are not supported but are the de facto standard expected by yarpOpenPose users.
        # For compatibility reasons we report them here and we add these keypoints with zero confidence to the output of e2pose.
        self._body25_keypoints = {"Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"}
        self.absent_keypoints = set(self._body25_keypoints) - set(self.coco_keypoints.values())

        return True

    def cleanup(self):
        self.in_port_image.close()
        self.out_port_image.close()
        self.out_port_target.close()
        return True

    def interruptModule(self):
        self.in_port_image.close()
        self.out_port_image.close()
        self.out_port_target.close()
        return True

    def updateModule(self):

        received_image = self.in_port_image.read()
        self.in_buf_image.copy(received_image)

        assert self.in_buf_array.__array_interface__['data'][0] == self.in_buf_image.getRawImage().__int__()

        #convert yarp to useful format 
        frame = np.copy(self.in_buf_array)

        #convert yarp to useful format 
        frame = cv2.resize(frame, dsize=(self.model_input_size[0],self.model_input_size[1]), interpolation=cv2.INTER_CUBIC)
        pred = self.model(np.stack([frame], axis=0))
        #TODO: pay attention as originally we decode a shape of raw not of frame
        humans = self.model.decode(pred, frame.shape[:2])

        target_bottle = yarp.Bottle()
        subtarget_bottle = yarp.Bottle() #This is added to preserve compatibility w/ yarpopenpose

        for human in humans:
            keypoints = human["keypoints"]
            keypoints = np.reshape(keypoints,[-1,3])

            human_bottle = yarp.Bottle()
            for i, keypoint in enumerate(keypoints):
                keypoint_bottle = yarp.Bottle()
                keypoint_bottle.addString(self.coco_keypoints[i]) #Keypoint tag ("nose","left eye", ...)
                keypoint_bottle.addFloat64(keypoint[0]) #x coordinate
                keypoint_bottle.addFloat64(keypoint[1]) #y coordinate
                keypoint_bottle.addFloat64(keypoint[2]) #confidence
                human_bottle.addList().read(keypoint_bottle)

            # Add fake neck positions and 0 confidence as it is not present in e2pose model
            
            for absent_keypoint in self.absent_keypoints:
                absent_keypoint_bottle = yarp.Bottle()
                absent_keypoint_bottle.addString(absent_keypoint)
                yarpE2PoseModule.fakeKeypoint(absent_keypoint_bottle)
                human_bottle.addList().read(absent_keypoint_bottle)

            # Add fake face keypoints
            face_keypoint_bottle = yarp.Bottle()
            face_keypoint_bottle.addString("Face")
            for _ in range(self.face_keypoints):
                fake_face_keypoint = yarpE2PoseModule.fakeKeypoint()
                face_keypoint_bottle.addList().read(fake_face_keypoint)
            human_bottle.addList().read(face_keypoint_bottle)

            subtarget_bottle.addList().read(human_bottle)

        target_bottle.addList().read(subtarget_bottle)

        image = self.painter(frame,humans)        
        image = cv2.resize(image, dsize=(self.image_width,self.image_height), interpolation=cv2.INTER_CUBIC)
        
        self.out_buf_array[:,:] = image
        self.out_port_image.write(self.out_buf_image)
        self.out_port_target.write(target_bottle)
        return True

    def getPeriod(self):
        return self.period

    @staticmethod
    def fakeKeypoint(keypoint_bottle=None):

        if keypoint_bottle is None:
            keypoint_bottle = yarp.Bottle()

        for _ in range(3):
            keypoint_bottle.addFloat32(0.0)
                
        return keypoint_bottle


if __name__ == '__main__':
    yarp.Network.init()

    if not yarp.Network.checkNetwork():
        print("yarpserver is not running")
        quit(-1)

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultConfigFile(os.path.join(yarpE2Pose_path,'app/conf/yarpE2Pose.ini'))
    rf.configure(sys.argv)

    manager = yarpE2PoseModule()
    manager.runModule(rf)
