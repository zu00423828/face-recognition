import cv2
import numpy as np
from numpy.linalg import norm
import mxnet
from glob import glob
from collections import namedtuple

from model.Symbol_MobileFace_Identification_V3 import *


class MobileFaceFeatureExtractor(object):
    def __init__(self, model_file, epoch, batch_size, context = mxnet.cpu()):
        self.model_file = model_file
        self.epoch = epoch
        self.batch_size = batch_size
        self.context = context

        network = get_feature_symbol_mobileface_v3() 
        self.model = mxnet.mod.Module(symbol = network, context = context)
        self.model.bind(for_training = False, data_shapes=[('data', (self.batch_size, 3, 112, 112))])
        sym, arg_params, aux_params = mxnet.model.load_checkpoint(self.model_file, self.epoch)
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
    def __init__(self,model_file,thresold):
        self.face_feature_extractor = MobileFaceFeatureExtractor(model_file, 0, 1, mxnet.cpu())
        self.thresold=thresold
    def get_feature(self,img):
        img=cv2.resize(img,(112,112))
        face_batch=[img]
        feature = self.face_feature_extractor.get_face_feature_batch(np.array(face_batch))
        feature[0]-=np.mean(feature[0])
        return feature[0]

    def get_cosine_similarity(self,default_feature,new_feature):
        # result=np.dot(default_feature, new_feature) / (norm(default_feature) * norm(new_feature)) # one to one 
        result=default_feature.dot(new_feature) / (norm(default_feature,axis=1) * norm(new_feature)) # many to one
        return result

    def compare_similarity(self,people_feature,img):
        compelte=[]
        img_feature = self.get_feature(img)
        cosine_similarity_scores=self.get_cosine_similarity(people_feature, img_feature)
        print(cosine_similarity_scores)
        for i,score in enumerate(cosine_similarity_scores):
            if score > self.thresold:
                compelte.append({'photoID':i,'confidence':score})
        return compelte

# def test_time():
#     facerecognition.compare_similarity(a1,a2)
if __name__ == "__main__":
    model_file = 'model/MobileFace_Identification_V3'
    facerecognition = FaceRecognition(model_file,0.5)
    a1=cv2.imread('test_data/a1.png')
    a2=cv2.imread('test_data/a2.png')
    b1=cv2.imread('test_data/b1.png')
    b2=cv2.imread('test_data/b2.png')
    c1=cv2.imread('test_data/c1.png')
    d1=cv2.imread('test_data/d1.png')



    people_feature=np.array([
        facerecognition.get_feature(a1),
        facerecognition.get_feature(b1),
        facerecognition.get_feature(b2),
        facerecognition.get_feature(c1),
        facerecognition.get_feature(d1),
    ])
    print(facerecognition.compare_similarity(people_feature,a2))


    # from timeit import timeit
    # t=timeit('test_time','from __main__ import test_time',number=1)
    # print(t)