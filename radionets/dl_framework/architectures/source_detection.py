from torch import nn
from radionets.dl_framework.model import Lambda, shape 


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(1,64, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv2 = nn.Sequential(
        nn.Conv2d(64,64, padding = 1, kernel_size = 3), nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = False)
        self.conv3 = nn.Sequential(
        nn.Conv2d(64,128, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv4 = nn.Sequential(
        nn.Conv2d(128,128, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv5 = nn.Sequential(
        nn.Conv2d(128,256, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv6 = nn.Sequential(
        nn.Conv2d(256,256, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv7 = nn.Sequential(
        nn.Conv2d(256,256, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv8 = nn.Sequential(
        nn.Conv2d(256,512, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv9 = nn.Sequential(
        nn.Conv2d(512,512, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv10 = nn.Sequential(
        nn.Conv2d(512,512, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv11 = nn.Sequential(
        nn.Conv2d(512,512, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv12 = nn.Sequential(
        nn.Conv2d(512,512, padding = 1, kernel_size = 3), nn.ReLU())
        self.conv13 = nn.Sequential(
        nn.Conv2d(512,512, padding = 1, kernel_size = 3), nn.ReLU(), nn.Dropout())
        self.fc1 = nn.Sequential(nn.Linear(512 * 9 * 9, 4096), nn.ReLU(), nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(4096, 4))   
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool(x)
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool(x)
        
        x = self.fc1(x.reshape(-1, 512 * 9 * 9))
        
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
