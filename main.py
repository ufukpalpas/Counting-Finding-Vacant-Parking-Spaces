from runOnImgOrVid import RunOnImgOrVid
from manual import Manual
from auto import Auto

def runAll():
    print("Please select to continue:")
    print("(0) Run for image")
    print("(1) Run for video without calc frames")
    print("(2) Run for video with calc frames (if you have frames as a pickle)")
    print("(3) Exit")
    val2 = input("Select:")
    if val2 == "0":
        runIMG = RunOnImgOrVid()
        runIMG.singleImageProcess(imageName="carParkImg.png", width=1100, height=720)
    elif val2 == "1":
        runVid = RunOnImgOrVid()
        frames = runVid.getFrames()
        h, w, _ = frames[0].shape
        runVid.processAndRunVideo(frames=frames, width=w, height=h)
    elif val2 == "2":
        runVid = RunOnImgOrVid()
        h, w = 720, 1100
        runVid.processAndRunVideo(frames=None, width=w, height=h)    
    elif val2 == "3":
        exit() 

if __name__ == '__main__': 
    print("Please select to continue:")
    print("(0) Run manual parking slot detection mode (if you have one you can edit too)")
    print("(1) Run automatic parking slot detection algorithm")
    print("(2) I have coordinates for car park, continue with determining vacant places")
    print("(3) Exit")
    print("Note: Please be careful about the hardcoded image/video sizes and names.")
    val = input("Select:")
    if val == "0":
        man = Manual()
        man.runman(imname="carParkImg.png")
        runAll()
    elif val == "1":
        aut = Auto()
        aut.runAuto("img1.jpg")
        print("Please select to continue:")
        print("(0) Run manual editting mode")
        print("(1) Continue without editing")
        print("(2) Exit")
        print("Note: You can see the outputs of automatic parking slot detection algorithm by running \"parkingSpots.ipynb\"")
        val3 = input("Select:")
        if val3 == "0":
            man = Manual()
            man.runman(imname="img1.jpg")
            runAll() 
        elif val3 == "1":
            runAll()
        elif val3 == "2":
            exit()
    elif val == "2":
       runAll()
    elif val == "3":
        exit()