import cv2
import numpy as np
from numpy.core.numeric import load
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
bboxes_predictor=None
landmark_predictor=None
align_tool=None

def load_model():
    global bboxes_predictor
    global landmark_predictor
    global align_tool
    bboxes_predictor = MobileFaceDetection('model/mobilefacedet_v1_gluoncv.params', '')
    landmark_predictor = dlib.shape_predictor('model/mobileface_landmark_emnme_v1.dat')
    align_tool = MobileFaceAlign('model/mobileface_align_v1.npy')

def face_align(img_path):
    landmark_num = 5
    if bboxes_predictor is None:
        load_model()
    align_size = (112, 112) 

    img_mat = cv2.imread(img_path)
    results = bboxes_predictor.mobileface_detector(img_path, img_mat)
    if results == None or len(results) < 1:
        raise Exception('not face')       
    for result in results:
        xmin, ymin, xmax, ymax, _, _ = result

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
        shape = landmark_predictor(img_mat, dlib_box)
        points = []
        for k in range(landmark_num):
            points.append([shape.part(k).x, shape.part(k).y])
        align_points = []
        align_points.append(points)
        align_result = align_tool.get_align(img_mat, align_points, align_size)
        return align_result[0]

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

class FaceRecognition():
    def __init__(self, model_file, threshold):
        self.face_feature_extractor = MobileFaceFeatureExtractor(
            model_file, 0, 1, mxnet.cpu())
        self.threshold = threshold

    def get_feature(self, img_paths):
        feature_list = []
        for item in img_paths:
            img = face_align(item)
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
        # dist = 1/(1+dist)
        # dist=norm(feature_1-feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist


    def compare_similarity(self, people_data, img):
        df = pd.DataFrame(people_data)
        people_name = df['filename'].tolist()
        people_feature = np.array(df['feature'].tolist())
        compelte = []
        img_feature = self.get_feature([img])[0]["feature"]
        # cosine_similarity_scores = self.get_cosine_similarity(
        #     people_feature, img_feature)
        # print(cosine_similarity_scores)
    
        for i, (person_name, person_feature) in enumerate(zip(people_name, people_feature)):
            score = self.get_euclidean_distance(person_feature, img_feature)
            print(person_name,score)
            if score < self.threshold:
                compelte.append({'photoID': person_name, 'confidence': 1-score})
        return compelte


def test_time():
    facerecognition.compare_similarity(people_feature,a2)
if __name__ == "__main__":
    model_file = 'model/MobileFace_Identification_V3'
    facerecognition = FaceRecognition(model_file, 0.5)
    a2 = 'test_data/101.png'
    filelist = ['test_data/3.png','test_data/101.png','test_data/0.png'
                ]
    people_feature = facerecognition.get_feature(filelist)

    print(facerecognition.compare_similarity(people_feature, a2))


    from timeit import timeit
    print()
    t=timeit('test_time','from __main__ import test_time',number=3)
    print('cost:',t)
