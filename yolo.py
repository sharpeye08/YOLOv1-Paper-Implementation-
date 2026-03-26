import torch
import torch.nn as nn

# helper block to avoid repetation
class ConvBlock(nn.Module):
    def __init__(self , in_channles , out_channels , kernel_size , strdie=1 , padding = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels , out_channels , kernel_size , stride, padding , bias= False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self ,x):
        return self.act(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self , S =7, B=2 , C =20):
        super().__init__()

        self.backbone = nn.Sequential(
            # block 1
            ConvBlock(3 , 64 , 7 , stride = 2 , padding = 3), 
            nn.MaxPool2d(2,2),

            # block 2
            ConvBlock(64 , 192 , 3, padding=1),
            nn.MaxPool2d(2 ,2),

            # block 3
            ConvBlock(192 , 128 , 1),
            ConvBlock(128 , 256 , 3 , padding= 1),
            ConvBlock(256 , 256 , 1),
            ConvBlock(256 , 512 , 3 , padding= 1),
            nn.MaxPool2d(2,2),

            # block 4
            ConvBlock(512 , 256 , 1),
            ConvBlock(256 , 512 , 3 ,padding=1),
            ConvBlock(512 , 256 , 1),
            ConvBlock(256 , 512 , 3 ,padding=1),
            ConvBlock(512 , 256 , 1),
            ConvBlock(256 , 512 , 3 ,padding=1),
            ConvBlock(512 , 256 , 1),
            ConvBlock(256 , 512 , 3 ,padding=1),
            ConvBlock(512 , 512 , 1),
            ConvBlock(512 , 1024 , 3 , padding= 1),
            nn.MaxPool2d(2 ,2),

            # block 5
            ConvBlock(1024 , 512 , 1),
            ConvBlock(512 , 1024 , 3 , padding= 1),
            ConvBlock(1024 , 512 , 1),
            ConvBlock(512 , 1024 , 3 , padding= 1),
            ConvBlock(1024 , 1024 , 3 , padding= 1),
            ConvBlock(1024 , 1024 , 3 ,strdie=2 , padding= 1),

            # block 6
            ConvBlock(1024 , 1024 , 3 , padding= 1),
            ConvBlock(1024 , 1024 , 3 , padding= 1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4046),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096 , S * S * (B * 5 + C)),
        )

        self.S = S
        self.B = B
        self.C = C

    def forward(self , x):
        x = self.backbone(x)
        x = self.fc(x)
        
        return x.view(-1 , self.S , self.S , self.B * 5 + self.C)
