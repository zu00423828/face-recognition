import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #disable gpu
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pickle
from keras.models import load_model
import cv2
import dlib
from tool.face_align import FaceAligner



detector = None
feature_extractor = None
fa = None


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
    predictor = dlib.shape_predictor(
        f'{model_dir}/shape_predictor_5_face_landmarks.dat')
    fa = FaceAligner(predictor)
    feature_extractor = load_model(f'{model_dir}/facenet_keras.h5')


def l2_normalize(x, axis=-1, epsilon=1e-10):

    output = x / np.sqrt(np.maximum(np.sum(np.square(x),
                         axis=axis, keepdims=True), epsilon))
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
        faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
        faceAligned = cv2.resize(faceAligned, (160, 160))
        return faceAligned
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def face_feature(img_path):
    img=face_align(img_path)
    white_img=prewhiten(img)
    white_img= white_img[np.newaxis,:]
    feature = l2_normalize(np.concatenate(feature_extractor.predict(white_img)))
    return feature

def knn_classifier(input, knn_path):
    knn = joblib.load(knn_path)
    x=face_feature(input)
    x=x.reshape(1,-1)
    labels=load_label_ids('labels.pkl')
    prob = knn.predict_proba(x)
    pred = knn.predict(x)
    print('pred',pred)
    print(labels[pred[0]])
    print(prob[0][pred])
    return pred, prob


def training_KNN(people_data):
    X_train= people_data['feature']
    Y_train =people_data['label']
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    knn.fit(X_train, Y_train)
    joblib.dump(knn, 'model/knn.model')
    # return print("---KNN model saved---")


def get_features_and_labels(image_paths, label_path='labels.pkl', sep='.'):
    x_train = []
    currect_id = 0
    label_ids = {}
    y_labels = []

    for  image_path in tqdm(image_paths):
        name = os.path.basename(image_path).split(sep)[0]
        # try:
        if name not in label_ids:
            label_ids[name] = currect_id
            currect_id += 1
        feature= face_feature(image_path)
        x_train.append(feature)
        y_labels.append(label_ids[name])
        # except Exception as e:
            # print(image_path, e)
    dump_label_ids(label_ids, label_path)


    return {'feature':x_train,'label':y_labels}


def dump_label_ids(label_ids, filename):
    if os.path.exists(filename):
        print('remove labels.pkl')
        os.remove(filename)
    with open(filename, 'wb') as f:
        pickle.dump(label_ids, f)



def load_label_ids(filename):
    label_ids = {}
    with open(filename, 'rb') as f:
        label_ids = pickle.load(f)
        # print(label_ids.keys())
        label_ids = {v: k for k, v in label_ids.items()}
    return label_ids


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default='face_data')
    parser.add_argument('--model_path', default='lbph_model.yml')
    parser.add_argument('--label_path', default='labels.pkl')
    parser.add_argument('--detector_path',
                        default='haarcascade_frontalface_default.xml')

    parser.add_argument('--test_img_path',
                        default='test_data/val_img/a_2.png')
    args = parser.parse_args()

    train_data_dir = args.train_data_dir
    test_img_path = args.test_img_path
    model_path = args.model_path
    label_path = args.label_path
    detector_path = args.detector_path
    print('args ', train_data_dir, test_img_path, model_path, detector_path)
    load_pretrain_model('model')


    image_paths=glob(f'{train_data_dir}/*')
    people_data = get_features_and_labels(image_paths, label_path)
    training_KNN(people_data)
    knn_classifier(test_img_path,'model/knn.model')

