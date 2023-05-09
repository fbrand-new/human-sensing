from yarpPose import PoseEstimation
from mmpose.apis import MMPoseInferencer

class yarpMMPose(PoseEstimation):

    def __init__(self,alias):
        self.inferencer = MMPoseInferencer(alias)

    def __init__(self,config,model):
        self.inferencer = MMPoseInferencer(config,model)

    def inference(self,img):
        return self.inferencer(img)

    # def configure():
    #     - configure the network

    # def updateModule()
    #     - leggi immagine
    #     - fai forward(immagine)
