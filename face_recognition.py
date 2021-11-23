import cv2
import numpy as np
from numpy.linalg import norm
import mxnet
from collections import namedtuple
import pandas as pd
import dlib
if __name__ == "__main__":
    from tool.Symbol_MobileFace_Identification_V3 import *
    from tool.MobileFace_Detection.mobileface_detector import MobileFaceDetection
    from tool.mobileface_alignment import MobileFaceAlign
else:
    from .tool.Symbol_MobileFace_Identification_V3 import *
    from .tool.MobileFace_Detection.mobileface_detector import MobileFaceDetection
    from .tool.mobileface_alignment import MobileFaceAlign

class MobileFaceFeatureExtractor(object):
    def __init__(self, model_file, epoch, batch_size, context=mxnet.cpu()):
        self.model_file = model_file
        self.epoch = epoch
        self.batch_size = batch_size
        self.context = context

        network = get_feature_symbol_mobileface_v3()
        self.model = mxnet.mod.Module(symbol=network, context=context)
        self.model.bind(for_training=False, data_shapes=[
                        ('data', (self.batch_size, 3, 112, 112))])
        _, arg_params, aux_params = mxnet.model.load_checkpoint(
            self.model_file, self.epoch)
        self.model.set_params(arg_params, aux_params)

    def get_face_feature_batch(self, face_batch):
        Batch = namedtuple('Batch', ['data'])
        batch_data = np.zeros((self.batch_size, 3, 112, 112))
        face_batch = face_batch.astype(np.float32, copy=False)
        face_batch = (face_batch - 127.5)/127.5
        batch_data = face_batch.transpose(0, 3, 1, 2)
        self.model.forward(Batch([mxnet.nd.array(batch_data)]))
        feature = self.model.get_outputs()[0].asnumpy().copy()
        return feature

class FaceRecognition():
    def __init__(self, model_dir, threshold):
        self.face_feature_extractor = MobileFaceFeatureExtractor(
            f'{model_dir}/MobileFace_Identification_V3', 0, 1, mxnet.cpu())
        self.bboxes_predictor = MobileFaceDetection(f'{model_dir}/mobilefacedet_v1_gluoncv.params', '')
        self.landmark_predictor = dlib.shape_predictor(f'{model_dir}/mobileface_landmark_emnme_v1.dat')
        self.align_tool = MobileFaceAlign(f'{model_dir}/mobileface_align_v1.npy')
        self.threshold = threshold
    def face_align(self,img_path): 
        '''
            get 112*112 face image is alignment
        '''
        align_size = (112, 112) 
        img_mat = cv2.imread(img_path)
        bboxes = self.bboxes_predictor.mobileface_detector(img_path, img_mat)
        if bboxes == None or len(bboxes) < 1:
            raise Exception('not face')       
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, _, _ = bbox
            size_scale = 0.75
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
            shape = self.landmark_predictor(img_mat, dlib_box).parts()
            points = [[p.x,p.y] for p in shape]
            align_result = self.align_tool.get_align(img_mat, [points], align_size)
            return align_result[0]
    def get_feature(self, img_paths):  
        '''
            get 1*256 face vetor
        '''
        feature_list = []
        for item in img_paths:
            img = self.face_align(item)
            face_batch = np.array([img])
            feature = self.face_feature_extractor.get_face_feature_batch(face_batch
                )
            feature[0] -= np.mean(feature[0])
            feature_list.append({'filename': item, 'feature': feature[0]})
        return feature_list

    def get_cosine_similarity(self, default_feature, new_feature):
        # result=np.dot(default_feature, new_feature) / (norm(default_feature) * norm(new_feature)) # one to one
        result = default_feature.dot(
            new_feature) / (norm(default_feature, axis=1) * norm(new_feature))  # many to one
        return result

    def get_euclidean_distance(self, feature_1, feature_2):
        # feature_1 = np.array(feature_1)
        # feature_2 = np.array(feature_2)
        # dist = 1/(1+dist)
        # dist=norm(feature_1-feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist


    def compare_similarity(self, people_data, img):
        '''
            people feature and new_img make similarity
        '''
        df = pd.DataFrame(people_data)
        people_name = df['filename'].tolist()
        people_feature = np.array(df['feature'].tolist())
        compelte = []
        img_feature = self.get_feature([img])[0]["feature"]
        # cosine_similarity_scores = self.get_cosine_similarity(
        #     people_feature, img_feature)
        # print(cosine_similarity_scores)
    
        for (person_name, person_feature) in zip(people_name, people_feature):
            dist = self.get_euclidean_distance(person_feature, img_feature)
            print(person_name,dist)
            if dist < self.threshold:
                compelte.append({'photoID': person_name, 'confidence': 1-dist})
        return compelte


def test_time():
    facerecognition.compare_similarity(people_data,a2)

if __name__ == "__main__":
    model_dir = 'model'
    facerecognition = FaceRecognition(model_dir, 0.5)
    a2 = 'test_data/101.png'
    filelist = ['test_data/3.png','test_data/101.png','test_data/0.png'
                ]
    people_data = facerecognition.get_feature(filelist)# get people feature

    print(facerecognition.compare_similarity(people_data, a2)) # similarity


    from timeit import timeit
    print()
    number=100000
    t=timeit('test_time','from __main__ import test_time',number=number)
    print('average_cost:',t/number)
