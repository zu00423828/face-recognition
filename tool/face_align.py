import numpy as np
import time
import cv2
import dlib
from MobileFace_Detection.mobileface_detector import MobileFaceDetection

bboxes_predictor = MobileFaceDetection('../model/mobilefacedet_v1_gluoncv.params', '')
landmark_predictor = dlib.shape_predictor('../model/mobileface_landmark_emnme_v1.dat')
def face_align(img): 
    bboxes = bboxes_predictor.mobileface_detector('', img)
    if bboxes == None or len(bboxes) < 1:
        raise Exception('not face')       
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, _, _ = bbox
        size_scale = 1
        center_scale = 0.1
        center_shift = (ymax - ymin) * center_scale
        w_new = (ymax - ymin) * size_scale
        h_new = (ymax - ymin) * size_scale
        x_center = xmin + (xmax - xmin) / 2
        y_center = ymin + (ymax - ymin) / 2 + center_shift
        x_min = int(x_center - w_new / 2)
        y_min = int(y_center - h_new / 2)
        x_max = int(x_center + w_new / 2)
        y_max = int(y_center + h_new / 2)
        dlib_box = dlib.rectangle(x_min, y_min, x_max, y_max)
        shape = landmark_predictor(img, dlib_box).parts()
        points =np.array([[p.x,p.y] for p in shape])
        print(points[0],points[1])
        eye_dist=np.linalg.norm(points[0]-points[1])/w_new
        lefteye2noise=np.linalg.norm(points[0]-points[2])/h_new
        righteye2noise=np.linalg.norm(points[1]-points[2])/h_new
        eye2noise_dist=np.linalg.norm(((points[0]+points[1])/2)-points[2])/h_new
        noise2leftmouth=np.linalg.norm(points[2]-points[3])/h_new
        noise2rghtitmouth=np.linalg.norm(points[3]-points[4])/h_new
        mouth_dist=np.linalg.norm(points[3]-points[4])/w_new
        print(eye_dist,lefteye2noise,righteye2noise,eye2noise_dist,noise2leftmouth,noise2rghtitmouth,mouth_dist)
def get_angle():
    np.linalg.norm()
st=time.time()
img=cv2.imread('../test_data/0.png')
face_align(img)
et=time.time()
print(et-st)