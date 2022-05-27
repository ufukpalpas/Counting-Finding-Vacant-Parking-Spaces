from __future__ import division
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import pickle
import operator

class Auto():

    def __init__(self):
        print("Automatic parking spot creatin started.")    
        
    def runAuto(self, image="img1.jpg"):
        test_images = [plt.imread(image)]
        #show_images(test_images)
        print("Select end point of long edge - interior point of long edge - second interior point of short edge")
        global points
        points = [] # end point of long edge - interior point of long edge - second interior point of short edge
        closeCount = 0
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.select_point)
        cv2.imshow('image',test_images[0])
        cv2.waitKey(0)     
        cv2.destroyAllWindows()
        self.longEdge = math.sqrt((points[1][1]-points[0][1])**2+(points[1][0]-points[0][0])**2) #width
        self.shortEdge = math.sqrt((points[2][1]-points[1][1])**2+(points[2][0]-points[1][0])**2) #height
        #print(longEdge, shortEdge)
        #print(points)
        gaus = list(map(self.dilation, test_images))
        #show_images(gaus)
        white_yellow_images = list(map(self.select_rgb_white_yellow, gaus))
        #show_images(white_yellow_images)
        print("White yellow filter and dilation applied.")
        gray_images = list(map(self.convert_gray_scale, white_yellow_images))
        #show_images(gray_images)
        edge_images = list(map(lambda image: self.detect_edges(image), gray_images))

        #show_images(edge_images)
        print("Edges detected.")
        
        # images showing the region of interest only
        roi_images = list(map(self.select_region, edge_images))

        #show_images(roi_images)
        list_of_lines = list(map(self.hough_lines, roi_images))
        
        print("Hough line detection completed.")
        #print(list_of_lines)
        line_images = []
        self.shortEdgeCoef = 1/2 if self.shortEdge > 20 else 1
        for image, lines in zip(test_images, list_of_lines):
            line_images.append(self.draw_lines(image, lines, zoomed=False)[0])
        
        #show_images(line_images)
        # images showing the region of interest only
        rect_images = []
        rect_coords = []
        rect_single = []
        div_rects = []
        #buffAdd = 12 if longEdge >= 50 else 3
        self.buffAdd = ((self.longEdge - 30) // 10) * 2.31 + 3
        gapAdd = ((self.shortEdge - 14) // 10) * 0.38 + 0.50
        #print("gap: ", gapAdd)
        for image, edge, lines in zip(test_images, edge_images, list_of_lines):
            new_image, rects ,twoOrOne , dividedImgs= self.identify_blocks(image, edge, lines)
            rect_images.append(new_image)
            rect_coords.append(rects)
            rect_single.append(twoOrOne)
            div_rects.append(dividedImgs)
        #show_images(rect_images)
        print("Parking blocks identified.")
        delineated = []
        spot_pos = []
        for image, rects, twoOrOne in zip(test_images, rect_coords, rect_single):
            new_image, spot_dict = self.draw_parking(image, rects, twoOrOne)
            delineated.append(new_image)
            spot_pos.append(spot_dict)
            
        #show_images(delineated)
        #print(len(spot_pos))
        print("Parking lines created.")
        self.final_spot_dict = spot_pos[0]

        #print(len(final_spot_dict))
        marked_spot_images = list(map(self.assign_spots_map, test_images))
        #show_images(marked_spot_images)

        positions = []
        for spot in self.final_spot_dict.keys():
            x1, y1, x2, y2 = spot
            positions.append((int(x1),int(y1)))
            
        with open('carPositions', 'wb') as f:
            pickle.dump(positions, f)
        with open('parkWidthHeight', 'wb') as f:
            pickle.dump([int(self.longEdge), int(self.shortEdge)], f)

        cv2.imshow("image", marked_spot_images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        print("Pickles saved. We recommend that rearrange some little parking slot position problems using manual mode's edit mode which can run automatically.")


    def show_images(self, images, cmap=None):
        cols = 2
        rows = (len(images)+1)//cols
        
        plt.figure(figsize=(15, 12))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i+1)
            # use gray scale color map if there is only one channel
            cmap = 'gray' if len(image.shape)==2 else cmap
            plt.imshow(image, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()
        
    def select_point(self, event,x,y,flags,param):
        global ix,iy
        if event == cv2.EVENT_LBUTTONDOWN: # captures left button down
            ix,iy = x,y
            #print(ix,iy)
            points.append([ix,iy])
            if len(points) == 3:
                cv2.destroyWindow('image')
                cv2.destroyAllWindows()

    def dilation(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel)

    def select_rgb_white_yellow(self, image): 
        # white color mask
        lower = np.uint8([120, 120, 120])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(image, lower, upper)
        # yellow color mask
        lower = np.uint8([190, 190,   0])
        upper = np.uint8([255, 255, 255])
        yellow_mask = cv2.inRange(image, lower, upper)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked = cv2.bitwise_and(image, image, mask = mask)
        return masked

    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        return cv2.Canny(image, low_threshold, high_threshold)

    def filter_region(self, image, vertices):
        """
        Create the mask using the vertices and apply it to the input image
        """
        mask = np.zeros_like(image)
        if len(mask.shape)==2:
            cv2.fillPoly(mask, vertices, 255)
        else:
            cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
        return cv2.bitwise_and(image, mask)

        
    def select_region(self, image):
        """
        It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
        """
        # first, define the polygon by vertices
        rows, cols = image.shape[:2]
        pt_1  = [cols*0.05, rows*0.95]
        pt_2 = [cols*0.05, rows*0.95]
        pt_3 = [cols*0.05, rows*0.95]
        pt_4 = [cols*0.05, rows*0.05]
        pt_5 = [cols*0.95, rows*0.05] 
        pt_6 = [cols*0.95, rows*0.95]
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
        return self.filter_region(image, vertices)

    def hough_lines(self, image):
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)

    def draw_lines(self, image, lines, color=[255, 0, 0], zoomed=False, thickness=2, make_copy=True):
        # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
        if make_copy:
            image = np.copy(image) # don't want to modify the original
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if not zoomed:
                    if abs(y2-y1) <=1 and abs(x2-x1) >= (self.shortEdge + 5) * self.shortEdgeCoef and abs(x2-x1) <= self.longEdge * 2: #40 - 70 /20-55 || büyük: 20-60
                        cleaned.append((x1,y1,x2,y2))
                        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                else:
                    if abs(y2-y1) <=1:
                        cleaned.append((x1,y1,x2,y2))
                        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        """
        for i in cleaned:
            for j in cleaned:
                if i != j:
                    if abs(i[1] - j[1]) <= 1 and abs(i[0] - j[0]) <= 1: # DELETE SO CLOSE ACCORDİNG TO Y
                        cleaned = [x for x in cleaned if x != i]
        
        for x in cleaned:
            cv2.line(image, (x[0], x[1]), (x[2], x[3]), color, thickness)
        """
        #print(" Number of lines detected: ", len(cleaned))
        return image, len(cleaned)

    def doOverlap(self, l1X, l1Y, r1X, r1Y, l2X, l2Y, r2X, r2Y):
        if(l1X == r1X or l1Y == r1Y or l2X == r2X or l2Y == r2Y):
            return False
        if(l1X >= r2X or l2X >= r1X):
            return False
        if(r1Y <= l2Y or r2Y <= l1Y):
            return False
        return True

    def identify_blocks(self, image, edge_image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(y2-y1) <=1 and abs(x2-x1) >= (self.shortEdge + 5) * self.shortEdgeCoef and abs(x2-x1) <= self.longEdge * 2: #20-55
                    cleaned.append((x1,y1,x2,y2))
        """
        for i in cleaned:
            for j in cleaned:
                if i != j:
                    if abs(i[1] - j[1]) <= 1 and abs(i[0] - j[0]) <= 1: # DELETE SO CLOSE ACCORDİNG TO Y
                        cleaned = [x for x in cleaned if x != i]
        """

        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))
        
        #Find clusters of x1 close together - clust_dist apart
        clusters = {}
        dIndex = 0
        clus_dist = self.shortEdge if self.shortEdge > 20 else self.shortEdge/2 + 1.5 #9 #15 #7
        #print("clus dist: ", clus_dist)

        for i in range(len(list1) - 1):
            distance = abs(list1[i+1][0] - list1[i][0])
            if distance <= clus_dist:
                if not dIndex in clusters.keys(): clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])

            else:
                dIndex += 1

        #Identify coordinates of rectangle around this cluster
        rects = {}
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 15:
                cleaned = sorted(cleaned, key=lambda tup: tup[1])
                avg_y1 = cleaned[0][1]
                avg_y2 = cleaned[-1][1]
        #         print(avg_y1, avg_y2)
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1/len(cleaned)
                avg_x2 = avg_x2/len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1
        """for rect in rects:
            print(rects[rect][2] - rects[rect][0])"""
        
        popALL = []
        for rect in rects:
            for rect2 in rects:
                if rects[rect] != rects[rect2]:
                    overlap  = self.doOverlap(rects[rect][0],rects[rect][1],rects[rect][2],rects[rect][3],rects[rect2][0],rects[rect2][1],rects[rect2][2],rects[rect2][3])
                    if overlap:
                        nX= rects[rect][2]-rects[rect][0]
                        nY = rects[rect][1]
                        sSide = math.sqrt((nX-rects[rect][0])**2 + (nY-rects[rect][1])**2)
                        lSide = math.sqrt((rects[rect][2]-nX)**2 + (rects[rect][3]-nY)**2)
                        area = sSide * lSide
                        nX2= rects[rect2][2]-rects[rect2][0]
                        nY2 = rects[rect2][1]
                        sSide2 = math.sqrt((nX2-rects[rect2][0])**2 + (nY2-rects[rect2][1])**2)
                        lSide2 = math.sqrt((rects[rect2][2]-nX2)**2 + (rects[rect2][3]-nY2)**2)
                        area2 = sSide2 * lSide2
                        
                        if area > area2:
                            popALL.append(rect2)
                        else:
                            popALL.append(rect)

        for i in range(len(popALL)):
            rects.pop(popALL[i],None)

        print("Num Parking Lanes: ", len(rects))
        
        #Draw the rectangles
        buff = self.shortEdge + self.buffAdd
        imgs = []
        imgsNorm = []
        imgCoords = []
        imgsNormAll = []
        for key in rects:
            img = edge_image[int(rects[key][1]):int(rects[key][3]), int(rects[key][0] - buff):int((rects[key][2] + rects[key][0])/2)]
            img2 = edge_image[int(rects[key][1]):int(rects[key][3]), int((rects[key][2] + rects[key][0])/2):int(rects[key][2] + buff)]
            imgNorm = image[int(rects[key][1]):int(rects[key][3]), int(rects[key][0] - buff):int((rects[key][2] + rects[key][0])/2)]
            imgNorm2 = image[int(rects[key][1]):int(rects[key][3]), int((rects[key][2] + rects[key][0])/2):int(rects[key][2] + buff)]
            imgCoords.append([int(rects[key][0] - buff),int(rects[key][1]), int((rects[key][2] + rects[key][0])/2),int(rects[key][3])])
            imgCoords.append([int((rects[key][2] + rects[key][0])/2),int(rects[key][1]), int(rects[key][2] + buff),int(rects[key][3])])
            imgs.append(img)
            imgs.append(img2)
            imgsNorm.append(imgNorm)
            imgsNorm.append(imgNorm2)
            imgNormAll = image[int(rects[key][1]):int(rects[key][3]), int(rects[key][0] - buff):int(rects[key][2] + buff)]
            imgsNormAll.append(imgNormAll)

        list_of_lines_in_lot = list(map(self.hough_lines, imgs))
        #print(list_of_lines_in_lot)
        line_images_in_lot = []
        num_of_lines = []
        for image, lines in zip(imgsNorm, list_of_lines_in_lot):
            temp_img, num_of = self.draw_lines(image, lines, zoomed=True)
            line_images_in_lot.append(temp_img)
            num_of_lines.append(num_of)
        #show_images(line_images_in_lot)
        #print(num_of_lines)

        twoOrOne = []
        for x in range(len(num_of_lines)):
            if num_of_lines[x] <= 1:
                twoOrOne.append(True)
                for rect in rects:
                    isInRect = self.doOverlap(rects[rect][0]-buff,rects[rect][1],rects[rect][2]+buff,rects[rect][3],imgCoords[x][0],imgCoords[x][1],imgCoords[x][2],imgCoords[x][3])
                    #print("rect: ", rects[rect][0]-buff,rects[rect][1],rects[rect][2]+buff,rects[rect][3])
                    #print("img: ", imgCoords[x][0],imgCoords[x][1],imgCoords[x][2],imgCoords[x][3])
                    #print(isInRect)
                    if isInRect:
                        if rects[rect][0] == imgCoords[x][0]:
                            rects[rect] = (imgCoords[x][0], imgCoords[x][1], rects[rect][2], rects[rect][3])
                        else:
                            rects[rect] = (rects[rect][0], rects[rect][1], (rects[rect][2] + rects[rect][0] + buff)/2 , rects[rect][3]) # -buff kaldırdık
            else:
                twoOrOne.append(False)
        tf_rects = []
        for i in range(0, len(twoOrOne), 2):
            if twoOrOne[i] or twoOrOne[i+1]:
                tf_rects.append(True)
            else:
                tf_rects.append(False)

        #print(twoOrOne)
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))   
    #        print(tup_topLeft, tup_botRight)
            cv2.rectangle(new_image, tup_topLeft,tup_botRight,(0,255,0),3)
        #print("tf_rects: ", tf_rects)
        return new_image, rects, tf_rects, imgsNormAll

    def draw_parking(self, image, rects, twoOrOne, make_copy = True, color=[255, 0, 0], thickness=2, save = True):
        if make_copy:
            new_image = np.copy(image)    
        gap = int(self.shortEdge)  #12.5 #15.5 # +0,5
        spot_dict = {} # maps each parking ID to its coords
        tot_spots = 0
        buff = int(self.shortEdge) + int(self.buffAdd) # 3
        avg_rect_size = 0

        #print(len(twoOrOne))

        for key in rects:
            h = rects[key][3] - rects[key][1]
            w = rects[key][2] - rects[key][0]
            avg_rect_size += h * w
        avg_rect_size = avg_rect_size / len(rects)
        count = 0
        for key in rects:     
            # Horizontal lines
            tup = rects[key]
            x1 = int(tup[0]) - buff#+ adj_x1[key])
            x2 = int(tup[2]) + buff#+ adj_x2[key])
            y1 = int(tup[1])# + adj_y1[key])
            y2 = int(tup[3])# + adj_y2[key])
            
            cv2.rectangle(new_image, (x1, y1),(x2,y2),(0,255,0),2)

            num_splits = int(abs(y2-y1)//gap)
            gapPlus = 0
            for i in range(0, num_splits):
                if num_splits//2 + 1 == i and gapPlus == 0:
                    gapPlus += 5
                y = int(y1 + i*gap + gapPlus)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
        
            #draw vertical lines
            if not twoOrOne[count]:
                x = int((x1 + x2)/2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
                
            # Add up spots in this lane
            if twoOrOne[count]: 
                tot_spots += num_splits 
            else:
                tot_spots += 2*(num_splits )
                
            # Dictionary of spot positions
            if twoOrOne[count]: #if area < avg_rect_size:
                gapPlus = 0
                for i in range(0, num_splits):
                    if num_splits//2 + 1 == i and gapPlus == 0:
                        gapPlus -= 4
                    cur_len = len(spot_dict)
                    y = (y1 + i*gap + gapPlus)
                    spot_dict[(x1, y, x2, y+gap + gapPlus)] = cur_len +1        
            else:
                gapPlus = 0
                for i in range(0, num_splits):
                    if num_splits//2 + 1 == i and gapPlus == 0:
                        gapPlus -= 2
                    cur_len = len(spot_dict)
                    y = y1 + i*gap + gapPlus
                    x = (x1 + x2)/2
                    spot_dict[(x1, y, x, y+gap + gapPlus)] = cur_len +1
                    spot_dict[(x, y, x2, y+gap + gapPlus)] = cur_len +2
            count += 1

        print("total parking spaces: ", tot_spots, cur_len)
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)
        return new_image, spot_dict

    def assign_spots_map(self, image, make_copy = True, color=[255, 0, 0], thickness=2):
        spot_dict = self.final_spot_dict
        if make_copy:
            new_image = np.copy(image)
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
        return new_image