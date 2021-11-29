import os
import dlib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #disable gpu
import time
import cv2
import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import distance
from keras.models import load_model

if __name__== '__main__':
    from tool.face_align import FaceAligner
else:
    from .tool.face_align import FaceAligner
image_size = 160
detector=None
fa=None
feature_extractor = None
def load_pretrain_model(model_dir):
    '''
    load 預訓練的模型 
    包含   dlib 的 hog 模型
    keras facemet 用於萃取特徵
    '''
    global detector
    global feature_extractor
    global fa
    # bboxes_predictor = MobileFaceDetection(f'{model_dir}/mobilefacedet_v1_gluoncv.params', '')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor( f'{model_dir}/shape_predictor_5_face_landmarks.dat')
    fa=FaceAligner(predictor)
    feature_extractor = load_model(f'{model_dir}/facenet_keras.h5')
def l2_normalize(x, axis=-1, epsilon=1e-10):

    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
def prewhiten(x):
    '''
    圖片白化
    解決過曝 或低曝圖片
    '''
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

def face_align(filename):
    '''
    人臉校正
    校正成眼睛水平
    resize成160*160
    '''
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for face in faces:
        faceAligned = fa.align(img, gray, face)
        faceAligned=cv2.cvtColor(faceAligned,cv2.COLOR_BGR2RGB)
        faceAligned=cv2.resize(faceAligned,(160,160))
        return faceAligned
    img=cv2.resize(img,(160,160))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
class FaceRecognition():
    '''
    參數
    model_dir \n
    threshold \n
    sep \n
    說明 \n
    model_dir 是放需要的模型資料夾\n
    threshold 是信心閥值 愈低愈好 預設為0.7 \n
    sep 是取得feature的檔案分割符號 用於做 label
    '''
    def __init__(self, model_dir='model', threshold=0.7,sep='.'):
        self.threshold = threshold
        self.sep=sep
        if detector is None:
            load_pretrain_model(model_dir)
    def get_feature(self, img_paths):  
        '''
            get 128 face vetor
        '''
        from tqdm import tqdm
        feature_list = []
        for item in tqdm(img_paths):
            try:
                name=os.path.basename(item).split(self.sep)[0]
                img=face_align(item)
                white_img=prewhiten(img)
                white_img= white_img[np.newaxis,:]
                feature = l2_normalize(np.concatenate(feature_extractor.predict(white_img)))
                feature_list.append({'label': name, 'feature': feature})
            except Exception as  e:
                print(item,e)
        # print(feature_list)
        return feature_list
    def compare_similarity(self, people_data, img_path):
        '''
            people feature and new_img make similarity
        '''
        df = pd.DataFrame(people_data)
        people_name = df['label'].tolist()
        people_feature = np.array(df['feature'].tolist())
        compelte = []
        try:
            img_feature = self.get_feature([img_path])[0]['feature']

            # print(cosine_similarity_scores)
            # print('min',np.min(cosine_similarity_scores),'max',np.max(cosine_similarity_scores))
            min_dist=100
            filename=''
            for (person_name, person_feature) in zip(people_name, people_feature):
                # dist = np.linalg.norm(person_feature.reshape(-1,1) - img_feature.reshape(-1,1),ord=2)
                dist=distance.euclidean(person_feature, img_feature)
                # print(person_name,dist)
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
        return img,compelte


if __name__ == "__main__":
    from glob import glob
    from random import shuffle
    from time import time
    model_dir = 'model'
    facerecognition = FaceRecognition(model_dir = model_dir, threshold = 0.7, sep = '.')
    filelist=glob('face_data/*')#face_data/*')
    shuffle(filelist)


    #取得 feature 和 label
    people_data = facerecognition.get_feature(filelist[:500])# get people feature



    from glob import glob

    # facerecognition.compare_similarity(people_data, 'test_data/e1.png')
    img_list=glob('test_data/*')
    img_list.sort()
    for img in img_list:
        st=time()
        #取得 辨識結果
        print(facerecognition.compare_similarity(people_data,  img)) # similarity
        et=time()
        print('cost:',f'{et-st:0.8f} s')
        print()

    # from timeit import timeit
    # print()
    # number=100000
    # t=timeit('test_time','from __main__ import test_time',number=number)
    # print('average_cost:',t/number)
