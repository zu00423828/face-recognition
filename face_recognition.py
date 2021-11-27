import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #disable gpu
from keras import backend as K
import time
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import distance
from keras.models import load_model
from tool.MobileFace_Detection.mobileface_detector import MobileFaceDetection
image_size = 160

bboxes_predictor = MobileFaceDetection('model/mobilefacedet_v1_gluoncv.params', '')
model_path = 'model/facenet_keras.h5'
model = load_model(model_path)
def l2_normalize(x, axis=-1, epsilon=1e-10):

    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

    return output
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y
def align_image(img):
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
        ymin=max(0,y_min)
        y_max=min(ymax,img.shape[0])
        x_min=max(0,x_min)
        y_max=min(ymax,img.shape[0])
        x_max=min(x_max,img.shape[1])
        face=img[y_min:y_max,x_min:x_max]
        aligned=cv2.resize(face,(160,160))
        return aligned
            
    # cascade = cv2.CascadeClassifier(cascade_path)

    # faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

    # if(len(faces)>0):
    #     (x, y, w, h) = faces[0]
    #     face = img[y:y+h, x:x+w]
    #     faceMargin = np.zeros((h+margin*2, w+margin*2, 3), dtype = np.uint8)
    #     faceMargin[margin:margin+h, margin:margin+w] = face
    #     aligned = cv2.resize(faceMargin, (image_size, image_size))
    #     return aligned
    # else:
    #     print('is not face')
    #     return None

class FaceRecognition():
    def __init__(self, model_dir, threshold):

        self.threshold = threshold

    def get_feature(self, img_paths):  
        '''
            get 128 face vetor
        '''
        from tqdm import tqdm
        feature_list = []
        for item in tqdm(img_paths):
            img=cv2.imread(item)
            # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            try:
                img=align_image(img)
                white_img=prewhiten(img)
                white_img= white_img[np.newaxis,:]
                feature = l2_normalize(np.concatenate(model.predict(white_img)))
                feature_list.append({'filename': item, 'feature': feature})
            #     img=cv2.imread(item)
            #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            #     for (x, y, w, h) in faces:
            #         x1 = x-PADDING
            #         y1 = y-PADDING
            #         x2 = x+w+PADDING
            #         y2 = y+h+PADDING
            #         img = cv2.rectangle(img,(x1, y1),(x2, y2),(255,0,0),2)
            #         feature = img_path_to_encoding(item, self.FRmodel)

            #     feature_list.append({'filename': item, 'feature': feature})
            except Exception as  e:
                print(item,e)
        # print(feature_list)
        return feature_list




    def compare_similarity(self, people_data, img):
        '''
            people feature and new_img make similarity
        '''
        df = pd.DataFrame(people_data)
        people_name = df['filename'].tolist()
        people_feature = np.array(df['feature'].tolist())
        compelte = []
        try:
            img_feature = self.get_feature([img])[0]['feature']

            # print(cosine_similarity_scores)
            # print('min',np.min(cosine_similarity_scores),'max',np.max(cosine_similarity_scores))
            min_dist=100
            filename=''
            for (person_name, person_feature) in zip(people_name, people_feature):
                # dist = np.linalg.norm(person_feature.reshape(-1,1) - img_feature.reshape(-1,1),ord=2)
                dist=distance.euclidean(person_feature, img_feature)
                print(person_name,dist)
                if dist < min_dist:
                    min_dist=dist
                    filename=person_name
                if  dist <self.threshold :
                    compelte.append({'photoID': person_name, 'confidence': dist})
                # compelte.append({'photoID': person_name, 'confidence': dist})
        except Exception as e:
            print('compare img is not face')
            return []
        print('min_dist',filename,min_dist)
        return compelte


def test_time():
    facerecognition.compare_similarity(people_data,filelist[-1])

if __name__ == "__main__":
    from glob import glob
    from random import shuffle
    from time import time
    model_dir = 'model'
    facerecognition = FaceRecognition(model_dir, 0.7)
    filelist=glob('test_data/face_data/*')#face_data/*')
    shuffle(filelist)
    people_data = facerecognition.get_feature(filelist[:500])# get people feature
    print('comparefile','test_data/101.png')
    print(people_data[0]['feature'].shape)
    st=time()
    # facerecognition.compare_similarity(people_data, 'test_data/e1.png')
    print(facerecognition.compare_similarity(people_data,  'test_data/101.png')) # similarity
    et=time()
    print('cost:',f'{et-st:0.8f} s')

    # from timeit import timeit
    # print()
    # number=100000
    # t=timeit('test_time','from __main__ import test_time',number=number)
    # print('average_cost:',t/number)
