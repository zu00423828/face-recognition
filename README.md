# import
```
from face_recognition import  FaceRecognition
```

# Init Object
```
facerecognition = FaceRecognition(model_dir, threshold, sep)
```
model_dir預設為None ，model_dir 用來讀取尋找人臉跟萃取特徵的模型資料夾， 可以依照自己的model_dir在哪 給參數 ，如果是None會讀 此專案的model資料夾
threshold預設為0.5 是信心值門檻
sep用於做label的切割符號
# Extract Feature
```
filelist=glob('face_data/*')#face_data/*')
people_data = facerecognition.get_feature(data_dir))# get people feature
```
data_dir 資料集資料夾

output
```
[{label:'xxx',featue:....},{label:'xxx',featue:....},{label:'xxx',featue:....}]
```

# Compare Similarity
```
facerecognition.compare_similarity(people_feature,img_path)
```

people_feature是 萃取所有已知的人臉feature的集合體

img_path是要被辨識的圖片路徑


# Example
```
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
        #取得 辨識結果
        print(facerecognition.compare_similarity(people_data,  img)) # similarity


```
```
[{'photoID': 0, 'confidence': 0.53109556}, {'photoID': 3, 'confidence': 0.51740336}]
```
