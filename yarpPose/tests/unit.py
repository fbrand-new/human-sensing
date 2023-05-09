import unittest
import cv2
from yarpMMPose import yarpMMPose

# Too slow to be a unit test
class MMPoseInferenceTest(unittest.TestCase):

    def test_keypoints(self):
        img = cv2.imread('data/000000000785.jpg')
        inferencer = yarpMMPose('human')
        keypoints = inferencer.inference(img)

        #These next lines can vary wildyl depending on the framework. We need a to abstract this
        self.assertTrue('head' in keypoints.keys())
        self.assertTrue(3,keypoints['head'].size())

if __name__ == "__main__":
    unittest.main()
