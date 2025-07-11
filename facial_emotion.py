import cv2
import mediapipe as mp
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from collections import Counter

# Define the Bottleneck class for ResNet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion, eps=0.001, momentum=0.99)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        # Downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # Add identity
        x += identity
        x = self.relu(x)
        
        return x

# Conv2dSame class for same padding
class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv_layer_s2_same = Conv2dSame(num_channels, 64, 7, stride=2, groups=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*ResBlock.expansion, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def extract_features(self, x):
        x = self.relu(self.batch_norm1(self.conv_layer_s2_same(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride, bias=False, padding=0),
                nn.BatchNorm2d(planes*ResBlock.expansion, eps=0.001, momentum=0.99)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

# Define the LSTM model
class LSTMPyTorch(nn.Module):
    def __init__(self):
        super(LSTMPyTorch, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)        
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x

# Function to create ResNet50 model
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

# Image preprocessing for PyTorch
def pth_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def __init__(self):
            super(PreprocessInput, self).__init__()
        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img):
        ttransform = transforms.Compose([
            transforms.PILToTensor(),
            PreprocessInput()
        ])
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        img = ttransform(img)
        img = torch.unsqueeze(img, 0)
        return img
   
    return get_img_torch(fp)

# Normalize coordinates
def norm_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# Get face bounding box from MediaPipe face landmarks
def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = norm_coordinates(landmark.x, landmark.y, w, h)
        if landmark_px:
            idx_to_coors[idx] = landmark_px
    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:,0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:,1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:,0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:,1])
    (startX, startY) = (max(0, x_min), max(0, y_min))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
   
    return startX, startY, endX, endY

# Function to display FPS on frame
def display_FPS(img, text, margin=1.0, box_scale=1.0):
    img_h, img_w, _ = img.shape
    line_width = int(min(img_h, img_w) * 0.001)  # line width
    thickness = max(int(line_width / 3), 1)  # font thickness
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)
    font_scale = thickness / 1.5
    t_w, t_h = cv2.getTextSize(text, font_face, font_scale, None)[0]
    margin_n = int(t_h * margin)
    sub_img = img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
              img_w - t_w - margin_n - int(2 * t_h * box_scale): img_w - margin_n]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
    img_w - t_w - margin_n - int(2 * t_h * box_scale):img_w - margin_n] = cv2.addWeighted(sub_img, 0.5, white_rect, .5, 1.0)
    cv2.putText(img=img,
                text=text,
                org=(img_w - t_w - margin_n - int(2 * t_h * box_scale) // 2,
                     0 + margin_n + t_h + int(2 * t_h * box_scale) // 2),
                fontFace=font_face,
                fontScale=font_scale,
                color=font_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False)
    return img

# Dictionary mapping emotion indices to names
DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}