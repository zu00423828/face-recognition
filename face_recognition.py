import cv2 
import os 
import numpy as np 
from PIL import Image
import pickle

import dlib
from face_align import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)





def face_align(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    name=os.path.basename(filename)
    for face in faces:
        faceAligned = fa.align(img, gray, face)
        faces=detector(cv2.cvtColor(faceAligned,cv2.COLOR_BGR2GRAY))
        for face in faces:
            size_h,size_w=faceAligned.shape[:2]
            (x, y, w, h) = rect_to_bb(face)
            x1 = max(x,0) 
            x2 = min(x+w,size_w)
            y1 = max(y,0)
            y2 = min(y+h,size_h)
            faceAligned=faceAligned[y:y+h,x:x+w]
            print(x,y,w,h)
            cv2.imwrite(f'new/{name}',faceAligned)
            return faceAligned
    return None


def get_images_and_labels(data_dir,detector_path='haarcascade_frontalface_default.xml',label_path='labels.pkl',sep='.'):
    detector = cv2.CascadeClassifier(detector_path)
    x_train = []
    currect_id = 0
    label_ids = {}
    y_labels = []
    image_paths=[os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    for _, image_path in enumerate(image_paths):
        name = os.path.basename(image_path).split(sep)[0]
        try:
            if name not in label_ids:
                label_ids[name] = currect_id
                currect_id += 1
            # img=cv2.imread(image_path)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img=face_align(image_path)
            if img is not None:
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                x_train.append(gray)
            # img = Image.open(image_path).convert('L')
            # gray = np.array(img,dtype=np.uint8)
            # faces = detector.detectMultiScale(gray)
            # for (x, y, w, h) in faces:
            #     data=gray[y:y + h, x:x + w]
            #     data=cv2.resize(data,(128,128))
            #     x_train.append(data)
                y_labels.append(label_ids[name])
        except Exception as e:
            print(image_path,e)
    dump_label_ids(label_ids,label_path)
    return x_train, y_labels

def dump_label_ids(label_ids,filename):
    with open(filename,'wb') as f:
        pickle.dump(label_ids,f)

def load_label_ids(filename):
    label_ids={}
    with open(filename,'rb') as f:
        label_ids=pickle.load(f)
        labels={v:k for k,v in label_ids.items()}
    return labels


def train_LBPHFaceRecognizer(data_dir,detecor_path='haarcascade_frontalface_default.xml', 
    model_path='lbph_model.yml', label_path='labels.pkl', data_sep='.'):
    '''
    params:
        detector_path \n
        mdoel_path \n
        label_path \n

    -- info
        data_dir is train data dir \n
        detector_path is opencv face detector model \n
        model_dir is lbph model save path \n 
        label_path is data label dump path \n
        data_sep is train data img file group, split filneane. ex: aaa.jpg - > aaa
    '''
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = get_images_and_labels(data_dir,detecor_path,label_path,data_sep)
    recognizer.train(faces, np.array(Ids))
    recognizer.save(model_path)



class FaceRecongition():
    '''
    params:

        mdoel_path \n
        label_path \n
        detector_path \n

    -- info
        mdoel_path is lbhp model path \n
        label_path is record classifier label path \n
        detecor_path is opencv face dete \n

    -- using \n
        facerecognition=FaceRecongition(model_path, label_path) \n
        print(facerecogition(test_img_path)) \n
 

        if model retrain call  reload method\n
            facerecogition.reload()
    
    
    '''
    def __init__(self,model_path='lbph_model.yml',label_path='labels.pkl',detector_path='haarcascade_frontalface_default.xml'):
        self.detector = cv2.CascadeClassifier(detector_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(model_path)
        self.labels = load_label_ids(label_path)
        self.model_path = model_path
        self.label_path = label_path
    def reload(self):
        self.labels = load_label_ids(self.label_path)
        self.recognizer.read(self.model_path)
    def __call__(self,img_path):
        img=cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:
            data = gray[y:y + h, x:x + w]
            data = cv2.resize(data,(128,128))
            id, conf = self.recognizer.predict(gray[y:y + h, x:x + w])
            # if 30 < conf < 85:
            return {'photoID': self.labels[id], 'confidence': conf}
            # return {'photoID':'Unknow','confidence':0}
        return {'error_message':'not face'}

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_data_dir',default='face_data')
    parser.add_argument('--model_path',default='lbph_model.yml')
    parser.add_argument('--label_path',default='labels.pkl')
    parser.add_argument('--detector_path',default='haarcascade_frontalface_default.xml')

    parser.add_argument('--test_img_path',default='test_data/a2.png')
    args=parser.parse_args()

    train_data_dir=args.train_data_dir
    test_img_path =args.test_img_path
    model_path = args.model_path
    label_path = args.label_path
    detector_path=args.detector_path
    print(train_data_dir,test_img_path,model_path,detector_path)
    
    train_LBPHFaceRecognizer(train_data_dir,detector_path)
    # facerecogition = FaceRecongition(model_path, label_path,detector_path)

    # from glob import glob
    # for item in glob('test_data/*'):
    #     face_align(item)

    # #retrain


    # train_LBPHFaceRecognizer(train_data_dir,detector_path)
    # facerecogition.reload()
    # print(facerecogition(test_img_path))
