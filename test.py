import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import localizerVgg
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plot
import cv2
import os
import math
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from CARPK import CARPK

def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    #print(detected_peaks)

    return detected_peaks

if __name__ == '__main__':
    downsampling_ratio = 8 # Downsampling ratio
    test_dataset = CARPK('', 'test', train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    cm_jet = mpl.cm.get_cmap('jet')
    model = localizerVgg.localizervgg16(pretrained=True, dsr=downsampling_ratio)
    model.load_state_dict(torch.load('weights.pt'))

    model.eval()
    model.cuda()

    gi = 0
    gRi = 0
    ind = 0
    countFound = 0
    countAll = 0
    diff = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (im, GAM, numCar, fname) in enumerate(test_loader):
            id_= batch_idx
            print(fname[0])
            image = im.cuda()
            MAP = model(image)
            cMap = MAP[0,0,].data.cpu().numpy()
            cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())
            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2], im.shape[3]))])

            cMap[cMap < 0.05] = 0
            peakMAP = detect_peaks(cMap)


            arrX = np.where(peakMAP)[0]
            arrY = np.where(peakMAP)[1]

            print(np.sum(peakMAP))
            print(int(numCar[0]))
            fark = np.sum(peakMAP) - int(numCar[0])
            diff += fark
            total += int(numCar[0])
            gi = gi + abs(fark)
            gRi = gRi + fark*fark
            ind = ind + 1

            print(id_,'\t', np.sum(peakMAP), int(numCar[0]), '\tAE: ', abs(fark))

            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2] , im.shape[3]))])

            img = im.numpy()[0,]
            img = np.array(img)
            img = (img - img.min()) / (img.max() - img.min())
            plot.imshow(img.transpose((1, 2, 0)))

            M1 = MAP.data.cpu().contiguous().numpy().copy()
            M1_norm = (M1[0, ] - M1[0, ].min()) / (M1[0, ].max() - M1[0, ].min())
            a = upsampler(torch.Tensor(M1_norm))
            a = np.uint8(cm_jet(np.array(a)) * 255)
            locs = []
            if batch_idx > 0:
                from PIL import Image
                ima = Image.fromarray(a)
                peakMAP = np.uint8(np.array(peakMAP) * 255)
                peakI = Image.fromarray(peakMAP).convert("RGB")
                peakI = peakI.resize((1280,720))
                #cv2.imwrite("res1/heatmap-" + str(batch_idx) + ".bmp",a)
                peakk = cv2.resize(peakMAP,dsize=(1280,720),interpolation=cv2.INTER_AREA)
                kernel = np.ones((3, 3), np.uint8)
                erPeak = cv2.erode(peakk,kernel)
                for g in range(1):
                    erPeak = cv2.erode(erPeak,kernel)

                temp = []

                for j in range(erPeak.shape[0]):#H
                    for i in range(erPeak.shape[1]):#W
                        if(erPeak[j][i] != 0):
                            sr = str(i) + " " + str(j)
                            temp.append(sr)

                cv2.imwrite("res1/peakmap-" + str(fname) + ".bmp",erPeak)
                #print(temp)
                #np.savetxt(("res1/peak-" + str(batch_idx) + ".txt"),temp)
                """addr = "CARPK\\Annotations\\" + fname[0] + ".txt"
                file = open(addr, "r")
                lines = file.readlines()

                for strr in temp:
                    splited = strr.split()
                    x = int(strr[0])
                    y = int(strr[1])
                    for line in lines:
                        arr = line.split()
                        x1 = int(arr[0])
                        y1 = int(arr[1])
                        x2 = int(arr[2])
                        y2 = int(arr[3])
                        if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                            countFound += 1
                            break
                
                for line in lines:
                    countAll += 1"""

                    


                locs.append(temp)
                #ima.save("res1/heatmap-" + str(batch_idx) + ".bmp")
                #peakI.save("res1/peakmap-" + str(batch_idx) + ".bmp")
                # print(peakI.size)
                # print(ima.size)
                # plot.imshow(a)
                # plot.show()

                # plot.imshow(peakMAP)
                # plot.show()

        print('MAE:', gi / ind)
        print('RMSE:', math.sqrt(gRi/ind))
        print("count found:", diff, "count all", total)



