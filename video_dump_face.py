import cv2
from glob import glob
import re
import os
from tool.MobileFace_Detection.mobileface_detector import MobileFaceDetection
from tool.mobileface_alignment import MobileFaceAlign
import dlib
bboxes_predictor = MobileFaceDetection('model/mobilefacedet_v1_gluoncv.params', '')
landmark_predictor = dlib.shape_predictor('model/mobileface_landmark_emnme_v1.dat')
align_tool = MobileFaceAlign('model/mobileface_align_v1.npy')
def face_align(img_mat): 
    '''
        get 112*112 face image is alignment
    '''
    align_size = (112,112)
    # img_mat = cv2.imread(img_path)
    # filename=os.path.basename(img_path)
    bboxes = bboxes_predictor.mobileface_detector('video', img_mat)
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
        shape = landmark_predictor(img_mat, dlib_box).parts()
        points = [[p.x,p.y] for p in shape]
        align_result = align_tool.get_align(img_mat, [points], align_size)
        return align_result[0]
videolist=glob('video/*.mp4')
dump_dir='dump_video_frame'
for item in videolist[26:]:
    # name=re.match('\S+(?=_\d+.mp4)',item).group(0)
    name=os.path.basename(item).replace('.mp4','_dir')
    video=cv2.VideoCapture(item)
    dump_path=os.path.join(dump_dir,name)
    os.makedirs(dump_path,exist_ok=True)
    count=0
    while video.isOpened():
        ret,frame=video.read()
        if not ret:
            break
        resut=face_align(frame)
        save_path=os.path.join(dump_path,f'{count:05d}.png')
        cv2.imwrite(save_path,resut)
        count+=1
    video.release()
