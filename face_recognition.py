import cv2
import numpy as np
from numpy.linalg import norm
import mxnet
from collections import namedtuple
import pandas as pd
if __name__ == "__main__":
    from tool.Symbol_MobileFace_Identification_V3 import *
    # from mobileface_detector import MobileFaceDetection
    # from mobileface_alignment import MobileFaceAlign
else:
    from .tool.Symbol_MobileFace_Identification_V3 import *


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
        sym, arg_params, aux_params = mxnet.model.load_checkpoint(
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


class MobileFaceAlign(object):
    pass


class FaceRecognition():
    def __init__(self, model_file, threshold):
        self.face_feature_extractor = MobileFaceFeatureExtractor(
            model_file, 0, 1, mxnet.cpu())
        self.threshold = threshold

    def get_feature(self, img_paths):
        feature_list = []
        for i, item in enumerate(img_paths):
            img = cv2.imread(item)
            img = cv2.resize(img, (112, 112))
            face_batch = [img]
            feature = self.face_feature_extractor.get_face_feature_batch(
                np.array(face_batch))
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
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        # dist = 1/(1+dist)
        # dist=norm(feature_1-feature_2)
        return dist


    def compare_similarity(self, people_data, img):
        df = pd.DataFrame(people_data)
        people_name = df['filename'].tolist()
        people_feature = np.array(df['feature'].tolist())
        print('people_feature', people_feature.shape)
        compelte = []
        img_feature = self.get_feature([img])[0]["feature"]
        cosine_similarity_scores = self.get_cosine_similarity(
            people_feature, img_feature)
        print(cosine_similarity_scores)
    
        for i, (person_name, person_feature) in enumerate(zip(people_name, people_feature)):
            score = self.get_euclidean_distance(person_feature, img_feature)
            print(person_name,score)
            if score < self.threshold:
                compelte.append({'photoID': person_name, 'confidence': score})
        return compelte


# def test_time():
#     facerecognition.compare_similarity(a1,a2)
if __name__ == "__main__":
    model_file = 'model/MobileFace_Identification_V3'
    facerecognition = FaceRecognition(model_file, 0.5)
    a2 = 'test_data/101.png'
    filelist = ['test_data/3.png','test_data/101.png','test_data/0.jpg','test_data/0.png'
                ]
    people_feature = facerecognition.get_feature(filelist)

    print(facerecognition.compare_similarity(people_feature, a2))


    # from timeit import timeit
    # t=timeit('test_time','from __main__ import test_time',number=1)
    # print(t)
