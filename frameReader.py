import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import  torchvision.transforms as transforms
import cv2
from skimage import transform as sktransform
import math
from image_augmentation import *
import random
object_categories = ['car',]

class videoSet(data.Dataset):

    def __init__(self, root, set, frames):
        self.root = root
        self.frames = frames
        self.path_images = os.path.join(root, 'images')
        self.ids = []
        # for root,dirs, files in os.walk(self.path_images):
        #     for file in files:
        #         self.ids.append(file)
        cnt = 1
        for i in range(len(self.frames)):
            self.ids.append(str(cnt) + ".jpg")
            cnt += 1
        self.numOfFrames = len(self.ids)
        if self.numOfFrames == 2:
            self.numOfFrames = 1
        print('Video frame set=%s  number of frames=%d' % (set, self.numOfFrames))

    def preprocess(self, img, min_size=720, max_size=1280):
        H, W, C = img.shape
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        img = sktransform.resize(img, (int(H * scale), int(W * scale), C), mode='reflect', anti_aliasing=True)
        img = np.asarray(img, dtype=np.float32)
        return img

    def __getitem__(self, index):
        id_ = self.ids[index]
        #path = os.path.join(self.path_images, str(index))
        #img = Image.open(os.path.join(path + '.jpg')).convert('RGB')
        img = self.frames[index]
        img = np.asarray(img, dtype=np.float32)

        H, W, _ = img.shape

        img = self.preprocess(img, min_size=H, max_size=W)

        if img.ndim == 2:
            img = img[np.newaxis]
        else:
            img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        normalize = transforms.Normalize(mean=[0.39895892, 0.42411209, 0.40939609], std=[0.19080092, 0.18127358, 0.19950577])
        img = normalize(torch.from_numpy(img))

        return img, id_

    def __len__(self):
        return len(self.ids)