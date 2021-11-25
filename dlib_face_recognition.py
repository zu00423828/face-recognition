import numpy  as np
from skimage import io
import imutils
import cv2
import dlib
from glob import glob
import os
import sys

faces_data_path='resource'
img_name=sys.argv[1]
detector=dlib.get_frontal_face_detector()
shape_predictor=dlib.shape_predictor('model/shape_predictor_68_face_landmarks_GTX.dat')
face_rec_model=dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')
descriptors=[]
candidate=[]


# get all face feature
for file in glob(f'{faces_data_path}/*png'):
    candidate.append(os.path.basename(file))
    img=io.imread(file)
    dets=detector(img,1)
    print(len(dets),file)
    for k,d in enumerate(dets):
        shape=shape_predictor(img,d)
        face_descriptor=face_rec_model.compute_face_descriptor(img,shape)
        v=np.array(face_descriptor)
        descriptors.append(v)
print(candidate)
#get one feature

import time

st=time.time()
img=io.imread(img_name)
print(img_name)
dets=detector(img,1)
distance=[]
for k,d in enumerate(dets):
    shape=shape_predictor(img,d)
    face_descriptor=face_rec_model.compute_face_descriptor(img,shape)
    d_test=np.array(face_descriptor)
    cx1=d.left()
    y1=d.top()
    x2=d.right()
    y2=d.bottom()
for i in descriptors:
    dist_=np.linalg.norm(i-d_test)
    distance.append(dist_)
candidate_distance_dict=dict(zip(candidate,distance))
et=time.time()
print(candidate_distance_dict)
candidate_distance_dict_sorted=sorted(candidate_distance_dict.items(),key=lambda d:d[1])
print(candidate_distance_dict_sorted)
result=candidate_distance_dict_sorted[0][0]
print(result)
print(f'{et-st:0.5f}s')