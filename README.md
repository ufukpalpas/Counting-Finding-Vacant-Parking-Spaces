# Counting-Finding-Vacant-Parking-Spaces
A drone view based vacant parking lot finding tool with VGG-16.

To run the code you may need to download required packages.
<br>
You can run the video by using the command 
```
python main.py
```
<br>
You need to download CARPK dataset and put the folders "Annotations" "Images" "ImageSets" folders under CARPK folder. (Dataset is 2GB so we cannot upload)
<br>
You can access and download dataset from: https://lafi.github.io/LPN/
<br>
You can try all parts of the program by using main.py file.
Image names are predifined in the code. you may need to change it if you want to try with different named videos or images.
The uploaded filter contains some pickles that contain information from previous runs. We leave it for you to see results without dealing with processing again.
If a new process is not started those locations will be protected.
<br>
The original source code of the train can be accessed from: <br>
https://github.com/ekilic/Heatmap-Learner-CNN-for-Object-Counting <br>
Our train code is an modified version of the code in the aforementioned repository.

If you don't want to train again you can simply use the weights given in above reporsitory by changing its name as "weights.py"
