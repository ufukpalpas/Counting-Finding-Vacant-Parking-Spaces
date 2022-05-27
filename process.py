import torch
import torchvision.transforms as transforms
import localizerVgg
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plot
import cv2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from frameReader import videoSet
import pickle

class processVid():
    def __init__(self, frames):
        self.frames = frames

    def detect_peaks(self, image):
        neighborhood = generate_binary_structure(2, 2)
        local_max = maximum_filter(image, footprint=neighborhood) == image
        background = (image == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        detected_peaks = local_max ^ eroded_background
        #print(detected_peaks)
        return detected_peaks

    def processVideo(self, width=1280, height=720):
        downsampling_ratio = 8 # Downsampling ratio
        test_dataset = videoSet('', 'test', self.frames)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        cm_jet = mpl.cm.get_cmap('jet')
        model = localizerVgg.localizervgg16(pretrained=True, dsr=downsampling_ratio)
        model.load_state_dict(torch.load('weights.pt'))
        model.eval()
        model.cuda()

        with torch.no_grad():
            locs = []
            for batch_idx, (im, fname) in enumerate(test_loader):
                id_= batch_idx
                image = im.cuda()
                MAP = model(image)
                cMap = MAP[0,0,].data.cpu().numpy()
                cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())

                upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2], im.shape[3]))])

                cMap[cMap < 0.05] = 0
                peakMAP = self.detect_peaks(cMap)

                upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2] , im.shape[3]))])

                img = im.numpy()[0,]
                img = np.array(img)
                img = (img - img.min()) / (img.max() - img.min())
                plot.imshow(img.transpose((1, 2, 0)))

                M1 = MAP.data.cpu().contiguous().numpy().copy()
                M1_norm = (M1[0, ] - M1[0, ].min()) / (M1[0, ].max() - M1[0, ].min())
                #a = upsampler(torch.Tensor(M1_norm))
                #a = np.uint8(cm_jet(np.array(a)) * 255)
                
                if batch_idx > 0:
                    from PIL import Image
                    #ima = Image.fromarray(a)
                    peakMAP = np.uint8(np.array(peakMAP) * 255)
                    peakI = Image.fromarray(peakMAP).convert("RGB")
                    peakI = peakI.resize((width,height))
                    #cv2.imwrite("res1/heatmap-" + str(batch_idx) + ".bmp",a)
                    peakk = cv2.resize(peakMAP,dsize=(width,height),interpolation=cv2.INTER_AREA) #interlinear
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

                    #cv2.imwrite("res1/peakmap-" + str(batch_idx) + ".bmp",erPeak) # to print peakmaps
                    print("Frame ", str(batch_idx), " completed")
                    #print(temp)
                    #np.savetxt(("res1/peak-" + str(batch_idx) + ".txt"),temp)
                    locs.append(temp)
                    #ima.save("res1/heatmap-" + str(batch_idx) + ".bmp")
                    #peakI.save("res1/peakmap-" + str(batch_idx) + ".bmp")
                    # print(peakI.size)
                    # print(ima.size)
                    # plot.imshow(a)
                    # plot.show()

                    # plot.imshow(peakMAP)
                    # plot.show()
                #if len(locs) > 5:
                    #return locs
                with open('carLocationsCalculated', 'wb') as f:
                    pickle.dump(locs, f)
            print('Processing completed!')
            return locs