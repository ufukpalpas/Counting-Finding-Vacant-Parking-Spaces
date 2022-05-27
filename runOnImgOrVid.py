import cv2
import pickle
from process import processVid

class RunOnImgOrVid():
    def __init__(self, fps=25, fname="carPark.mp4"):    
        self.interval = int(1000/fps) #FPS
        try:
            with open('carPositions', 'rb') as f:
                self.positions = pickle.load(f)
        except:
            print("Pickle containing parking slot coordinates should be created before!")
            exit()
        try: 
            with open('parkWidthHeight', 'rb') as f:
                self.sizeArr = pickle.load(f)
        except:
            print("Pickle containing parking slot width and height should be created before!")
            exit()
        self.width = int(self.sizeArr[0])
        self.height = int(self.sizeArr[1])
        self.capture = cv2.VideoCapture(fname)

    def isInParkingLot(self, poss, calcloc, width, height, allocSpaces ,ind):
        for loc in calcloc:
            xc = int(loc[0])
            yc = int(loc[1])
            xmin = poss[0]
            ymin = poss[1]
            xmax = poss[0] + width
            ymax = poss[1] + height
            #if ind == 0:
                #print("loc:", xc, yc,"rec", xmin, ymin, xmax, ymax )
            if(xmin <= xc) and (xc <= xmax) and (ymin <= yc) and (yc <= ymax):
                return True
        return allocSpaces
    
    def getFrames(self):
        frms = []
        count = 0
        print("Getting Frames...")
        while True:
            success, image = self.capture.read()
                
            # cv2.imshow("image", image)
            # cv2.waitKey(1)
            
            if success:
                #cv2.imshow("image", image)
                fileName = "images\\" + str(count) + ".jpg"
                #cv2.imwrite(fileName,image)
                frms.append(image)
                #if cv2.waitKey(self.interval) & 0xFF == ord('q'):
                #    break
            else:
                break
            count += 1
        cv2.destroyAllWindows()
        print("Frames are saved.")
        return frms
        
    def processAndRunVideo(self, frames=None, width=1280, height=720):
        if frames == None:
            with open('carLocationsCalculated', 'rb') as f:
                calculatedLocs = pickle.load(f)
        else:
            print("Processing Start...")
            vidProcess = processVid(frames)
            calculatedLocs = vidProcess.processVideo(width=width, height=height)
 
        calcLocs = []
        for fr in calculatedLocs:
            temp2 = []
            for l in range(len(fr)):
                temp1 = fr[l].split()
                temp2.append(temp1)
            calcLocs.append(temp2)       

        ct = 0
        while True:
            allocSpaces = []
            for t in range(len(self.positions)):
                allocSpaces.append(False)
                
            if self.capture.get(cv2.CAP_PROP_POS_FRAMES) == self.capture.get(cv2.CAP_PROP_FRAME_COUNT): #TO LOOP
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ct = 0
            success, image = self.capture.read()

            ind = 0
            if ct < len(calculatedLocs):
                for pos in self.positions:
                    allocSpaces[ind] = self.isInParkingLot(pos, calcLocs[ct], self.width, self.height, allocSpaces[ind], ct)
                    ind += 1
                    
            # for a in range(len(calcLocs[ct])):
            #     #print(calcLocs[ct][a][0], calcLocs[ct][a][1])
            #     cv2.circle(image, (int(calcLocs[ct][a][0]), int(calcLocs[ct][a][1])), radius=1, color=(0,0,255),thickness=-1)
            
            for pos, res in zip(self.positions, allocSpaces):
                if res:
                    cv2.rectangle(image,pos ,(pos[0] + self.width, pos[1] + self.height),(0,0,255),2)    
                else:
                    cv2.rectangle(image,pos ,(pos[0] + self.width, pos[1] + self.height),(0,255,0),2)    
            # cv2.imshow("image", image)
            # cv2.waitKey(1)
            
            if success:
                cv2.imshow("image", image)
                if cv2.waitKey(self.interval) & 0xFF == ord('q'):
                    break
            else:
                break
            ct += 1

        self.capture.release()
        cv2.destroyAllWindows()
        
    def singleImageProcess(self, imageName, width=1280, height=720):
        print("Getting İmage")
        image = cv2.imread(imageName)
        print("Processing...")
        frames = [image, image]
        vidProcess = processVid(frames)
        calculatedLocs = vidProcess.processVideo(width=width, height=height)

        calculatedLocsSingle = []
        for fr in calculatedLocs:
            temp2 = []
            for l in range(len(fr)):
                temp1 = fr[l].split()
                temp2.append(temp1)
            calculatedLocsSingle.append(temp2)  
        
        allocSpaces = []
        for t in range(len(self.positions)):
            allocSpaces.append(False)
        ct = 0   
        ind = 0
        for pos in self.positions:
            allocSpaces[ind] = self.isInParkingLot(pos, calculatedLocsSingle[0], self.width, self.height, allocSpaces[ind], ct)
            ind += 1
                    
        # for a in range(len(calcLocs[ct])):
        #     #print(calcLocs[ct][a][0], calcLocs[ct][a][1])
        #     cv2.circle(image, (int(calcLocs[ct][a][0]), int(calcLocs[ct][a][1])), radius=1, color=(0,0,255),thickness=-1)
            
        for pos, res in zip(self.positions, allocSpaces):
            if res:
                cv2.rectangle(image,pos ,(pos[0] + self.width, pos[1] + self.height),(0,0,255),2)    
            else:
                cv2.rectangle(image,pos ,(pos[0] + self.width, pos[1] + self.height),(0,255,0),2)    
        cv2.imshow("image", image)
        cv2.waitKey(0)   