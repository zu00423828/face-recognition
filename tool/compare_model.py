import torch
from torch import nn


class DistanceNet(nn.Module):
    def __init__(self):
        super(DistanceNet,self).__init__()
        self.distanse=nn.Sequential(nn.Linear(256,1),nn.Sigmoid())
    def forward(self,feature1,feature2):
        input=torch.abs(feature1-feature2)
        out=self.distanse(input)
        return out
if __name__=='__main__':
    moldel=DistanceNet()
    feature1=torch.ones(2,256)
    feature2=torch.zeros(2,256)
    out=moldel(feature1,feature2)
    print(out.shape)
    print(out)