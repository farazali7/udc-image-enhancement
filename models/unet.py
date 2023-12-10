import torch
from torch import nn



def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


########## BASIC UNET
class UNet(nn.Module):

    def __init__(self, n_class=3):
        super().__init__()
                
        self.down1 = conv_block(3, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.up3 = conv_block(256 + 512, 256)
        self.up2 = conv_block(128 + 256, 128)
        self.up1 = conv_block(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down3(x)
        x = self.maxpool(conv3)   
        
        x = self.down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.up1(x)
        
        out = self.conv_last(x)
        
        return out



##########  UNET Fusable NonConv
class UNet_fusable(nn.Module):

    def __init__(self, n_class=3):
        super().__init__()
                
        self.down1 = conv_block(3, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.up3 = conv_block(256 + 512, 256)
        self.up2 = conv_block(128 + 256, 128)
        self.up1 = conv_block(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x, f1=0, f2=0, f3=0, f4=0, f5=0, f6=0, f7=0):
        conv1 = self.down1(x)
        if self.training:        
            conv1 = conv1 + torch.mean(self.global_pool(f1))
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        if self.training:
            conv2 = conv2 + torch.mean(self.global_pool(f2))
        x = self.maxpool(conv2)
        
        conv3 = self.down3(x)
        if self.training:
            conv3 = conv3 + torch.mean(self.global_pool(f3))
        x = self.maxpool(conv3)   
        
        x = self.down4(x)
        if self.training:
            x = x + torch.mean(self.global_pool(f4))
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.up3(x)
        if self.training:
            x = x + torch.mean(self.global_pool(f5))
        
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)       

        x = self.up2(x)
        if self.training:
            x = x + torch.mean(self.global_pool(f6))

        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.up1(x)
        if self.training:
            x = x + torch.mean(self.global_pool(f7))
        
        out = self.conv_last(x)
        
        return out
