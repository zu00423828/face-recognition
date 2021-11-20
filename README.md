# import
```
from face_recognition import  FaceRecognition
```

# Init Object
```
facerecognition = FaceRecognition(model_file,thresold)
```
model_file預設為model/MobileFace_Identification_V3
threshold預設為0.5 是信心值門檻

# Extract Feature
```
facerecognition.get_feature(img_array)
```
img_array是 cv2讀取後的 np array
可使用 cv2.imread(path) 讀取



# Compare Similarity
```
facerecognition.compare_similarity(people_feature,other_img)
```

people_feature是 萃取所有已知的人臉feature的集合體

other_img是要被辨識的人臉


# Example
```
    from face_recognition import  FaceRecognition
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
```
```
[{'photoID': 0, 'confidence': 0.53109556}, {'photoID': 3, 'confidence': 0.51740336}]
```
