# import
```
from face_recognition import  FaceRecognition
```

# Init Object
```
facerecognition = FaceRecognition(model_file,threshold)
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
    facerecognition = FaceRecognition(model_file, 0.5)
    a2 = 'test_data/101.png'
    filelist = ['test_data/3.png','test_data/101.png','test_data/0.png'
                ]
    people_feature = facerecognition.get_feature(filelist)

    print(facerecognition.compare_similarity(people_feature, a2))
```
```
[{'photoID': 0, 'confidence': 0.53109556}, {'photoID': 3, 'confidence': 0.51740336}]
```
