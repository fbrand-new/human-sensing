from yarpPose import PoseEstimation
from mmpose.apis import MMPoseInferencer
import importlib.util
import sys

class yarpMMPose(PoseEstimation):

    def __init__(self,poseInferencer,dataset='COCO'):
        self.inferencer = poseInferencer
        self.dataset = dataset
   
    @classmethod
    def fromalias(cls,alias='human',dataset='COCO'):
        return cls(MMPoseInferencer(alias),dataset)
    
    @classmethod
    def fromconfig(cls,config,model,det_model=None,det_weights=None,dataset='COCO'):
        if det_model and det_weights:
            print(det_weights)
            inferencer = cls(MMPoseInferencer(config,model,det_model=det_model,det_weights=det_weights),dataset)
        else:
            inferencer = cls(MMPoseInferencer(config,model),dataset)

        return inferencer

    def inference(self,img):
        generator = self.inferencer(img)
        result = next(generator)

        if self.dataset == 'COCO':
            spec = importlib.util.spec_from_file_location("coco",'/mmpose/configs/_base_/datasets/coco.py')
            dataset_module = importlib.util.module_from_spec(spec)
            sys.modules["coco"] = dataset_module
            spec.loader.exec_module(dataset_module)

        keypoint_info = dataset_module.dataset_info["keypoint_info"]

        keypoints = [] 
        predictions = result["predictions"]

        for person in predictions[0]: 
            person_keypoints = {} 
            for i, keypoint in enumerate(person["keypoints"]):
                person_keypoints[keypoint_info[i]["name"]] = keypoint 
                person_keypoints[keypoint_info[i]["name"]].append(person["keypoint_scores"][i])

            keypoints.append(person_keypoints)
        
        return keypoints

    # def configure():
    #     - configure the network

    # def updateModule()
    #     - leggi immagine
    #     - fai forward(immagine)
