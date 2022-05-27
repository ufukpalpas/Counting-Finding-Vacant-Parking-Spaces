import cv2
import pickle

class Manual():
    def __init__(self):
        try:
            with open('carPositions', 'rb') as f:
                self.positions = pickle.load(f)
            print("manual edit mode active..")
        except:
           self.positions = []
        self.points = [] # left top coord - right bottom coord
        
    def select_point(self, event,x,y,flags,param):
        global ix,iy
        if event == cv2.EVENT_LBUTTONDOWN: # captures left button down
            ix,iy = x,y
            #print(ix,iy)
            self.points.append([ix,iy])
            if len(self.points) == 2:
                cv2.destroyWindow('image')
                cv2.destroyAllWindows()
                
    def clk(self, events, x, y, flags, params):
        if events == cv2.EVENT_LBUTTONDOWN:
            self.positions.append((x,y))
        if events == cv2.EVENT_RBUTTONDOWN:
            for i, pos in enumerate(self.positions):
                xg,yg = pos
                if xg < x < xg + self.width and yg < y < yg + self.height:
                    self.positions.pop(i)
        with open('carPositions', 'wb') as f:
            pickle.dump(self.positions, f)
        with open('parkWidthHeight', 'wb') as f:
            pickle.dump([self.width, self.height], f)
    
    def runman(self, imname = "carParkImg.png"):
        while True:
            image = cv2.imread(imname)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.select_point)
            cv2.putText(image, "1.click left top, 2. click right bottom", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('image',image)
            cv2.waitKey(0)     
            cv2.destroyAllWindows()
            image = cv2.imread(imname)
            cv2.putText(image, "You can see the size you selected, press \"c\" to continue, any other to reselect", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(image,(self.points[0][0], self.points[0][1]),(self.points[1][0], self.points[1][1]),(255,0,0),2)
            cv2.imshow("image", image)
            ky = cv2.waitKey(0)     
            cv2.destroyAllWindows()
            #print(self.points)
            if ky & 0xFF == ord("c"):
                break
            self.points = [] 
                
        self.width = self.points[1][0] - self.points[0][0]
        self.height = self.points[1][1] - self.points[0][1]

        while True: 
            image = cv2.imread(imname)
            cv2.putText(image, "Click top left of the area you want to select. Right Click to remove (q to exit)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            for pos in self.positions:
                cv2.rectangle(image,pos ,(pos[0] + self.width, pos[1] + self.height),(157,3,252),2)
            cv2.imshow("image", image)
            cv2.setMouseCallback('image', self.clk)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break