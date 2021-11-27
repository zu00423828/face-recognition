

import numpy as np
import argparse
import os
import pickle
import time
from glob import glob
import cv2
import face_recognition

from tool.dlib_mmod import detect as mmod_detect


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input", type=str, required=True, help="the input dataset path")
    ap.add_argument("-e", "--embeddings-file", type=str, required=False,default='embedding.npy',
                    help="the path to serialized db of facial embeddings")
    ap.add_argument("-d", "--detection-method", type=str, default="mmod", choices=["hog", "mmod"],
                    help="the detection method to use")
    args = vars(ap.parse_args())

    print("[INFO] loading dataset....")
    faces=[]
    names=[]
    for item in glob('test_data/*/*'):
        name=os.path.basename(item)
        img=cv2.imread(item)
        faces.append(img)
        names.append(name)
    faces=np.array(faces)
    names=np.array(names)

    # 由於Dlib處理圖片不同於OpenCV的BGR順序，需要先轉換成RGB順序
    faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]
    print(f"[INFO] {len(faces)} images in dataset")

    # 初始化結果
    known_embeddings = []
    known_names = []

    # # 建立我們的人臉embeddings資料庫
    # data = {}
    # print("[INFO] serializing embeddings...")
    if os.path.exists(args["embeddings_file"]):
        with open(args["embeddings_file"], "rb") as f:
            data = pickle.load(f)
    else:
        start = time.time()
        for (img, name) in zip(faces, names):
            # 偵測人臉位置
            rects = mmod_detect(img)

            # 將我們偵測的結果(x, y, w, h)轉為face_recognition使用的box格式: (top, right, bottom, left)
            boxes = [(rect[1], rect[0] + rect[2], rect[1] + rect[3], rect[0]) for rect in rects]
            print(boxes)
            embeddings = face_recognition.face_encodings(img, boxes)
            for embedding in embeddings:
                known_embeddings.append(embedding)
                known_names.append(name)

    #     print("[INFO] saving embeddings to file...")
        data = {"embeddings": known_embeddings, "names": known_names}
        with open(args["embeddings_file"], "wb") as f:
            pickle.dump(data, f)
        end = time.time()
        print(f"[INFO] serializing embeddings done, tooks {round(end - start, 3)} seconds")

    # 用已知的臉部資料庫來辨識測試資料集的人臉


    st=time.time()
    img=cv2.imread('e3.png')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #     # 這裡我們直接用face_recognition來偵測人臉
    boxes = face_recognition.face_locations(img, model="cnn")
    embeddings = face_recognition.face_encodings(img, boxes)
    print(len(embeddings))
    #     # 辨識結果
    names = []
    for embedding in embeddings:
        print(np.array(embeddings[0]).shape,np.array(data['embeddings']).shape)
        matches = face_recognition.compare_faces(data["embeddings"], embedding)
        name = "unknown"
        # matches是一個包含True/False值的list，會比對所有資料庫中的人臉embeddings
        print(matches)
        if True in matches:
            # 判斷哪一個人有最多matches
            matchedIdexs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdexs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
        names.append(name)
    print(names)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        # y = top - 8 if top - 8 > 8 else top + 8
        # cv2.putText(img, f"actual: {actual_name}", (left, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # cv2.putText(img, f"predict: {name}", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print(name)
    et =time.time()
    print(f"[INFO] face recognition, tooks {round(end - start, 3)} seconds")
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Result", img)
    # cv2.waitKey(0)


if __name__ == '__main__':
     main()