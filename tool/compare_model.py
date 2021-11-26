from numpy import array
import torch
from torch import nn
import mxnet
from collections import namedtuple

from torch._C import dtype
from Symbol_MobileFace_Identification_V3 import *
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

class DistanceNet(nn.Module):
    def __init__(self):
        super(DistanceNet,self).__init__()
        self.distanse=nn.Sequential(
            nn.Linear(512,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,1),nn.Sigmoid())
    def forward(self,feature1,feature2):
        input=torch.cat([feature1,feature2],dim=1)
        # input=torch.abs(feature1-feature2)
        out=self.distanse(input)
        return out
from torch.utils.data import Dataset
class DataSet(Dataset):
    def __init__(self,data_dir) :
        self.filelist=glob(f'{data_dir}/*/*')
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self,idx):
        origin=idx
        other=random.randint(0,len(self.filelist))
        if other == len(self.filelist):
            other=origin
        origin_img = cv2.imread(self.filelist[origin])
        other_img =cv2.imread(self.filelist[other])
        face_batch = np.array([origin_img,other_img])
        feature_batch = face_feature_extractor.get_face_feature_batch(face_batch)
        feature1=feature_batch[0]
        feature2=feature_batch[1]
        if (re.match('\S+(?=_\d+_dir)',os.path.dirname(self.filelist[origin])).group(0) ==
            re.match('\S+(?=_\d+_dir)',os.path.dirname(self.filelist[other])).group(0)):
            y=np.array([1],dtype=np.float32)
        else:
            y=np.array([0],dtype=np.float32)
        # y=np.array([1 if os.path.dirname(self.filelist[origin]) ==os.path.dirname(self.filelist[other]) else 0],dtype=np.float32)

        
        return feature1,feature2,y


def train():
    for epoch in range(max_epoch):
        bar=tqdm(dl)
        model.train()
        for (x1,x2,y) in bar:
            optimizer.zero_grad()
            y_pred=model(x1,x2)
            loss=loss_fuc(y_pred,y)
            loss.backward()
            optimizer.step()
            bar.set_description(f'Loss:{loss.item():0.5f}')
        torch.save(model,'checkpoint.pth')
        val()
def val():
    img1=cv2.imread('test_val/a/3.png')
    img2=cv2.imread('test_val/a/3.png')
    feature_batch = face_feature_extractor.get_face_feature_batch(np.array([img1,img2]))
    model.eval()
    with torch.no_grad():
        f1=torch.tensor(feature_batch[0]).unsqueeze(0)
        f2=torch.tensor(feature_batch[1]).unsqueeze(0)
        y_pred=model(f1,f2)
        print(y_pred)

    


if __name__=='__main__':
    from torch import optim
    from tqdm import tqdm
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import os
    from glob import glob
    import cv2
    import random
    import re 
    model=DistanceNet()
    dataset=DataSet('../dump_video_frame')#DataSet('LFW-Aligned-100Pair')
    
    optimizer=optim.Adam(model.parameters(),lr=1e-4)
    loss_fuc=nn.BCEWithLogitsLoss()
    batch_size=16
    max_epoch=200
    dl=DataLoader(dataset,batch_size=batch_size,drop_last=True,shuffle=True)
    face_feature_extractor = MobileFaceFeatureExtractor(
            '../model/MobileFace_Identification_V3', 0, 2, mxnet.cpu())
    model=torch.load('checkpoint.pth')
    train()
    # val()

    