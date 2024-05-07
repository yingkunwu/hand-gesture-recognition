# hand-gesture-recognition

Final project for UMN CSCI 5525 Machine Learning course

<img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/images/result.gif" alt="demo" width="500"/>
We only extract bounding boxes from the output of the hand detector. The classification task is performed by our model.

### Introduction
The objective of this project is to create a multi-task neural network which integrates pose estimation network with classification layers to maximize the performance of hand gesture recognition. I obtained good results by adding an additional layers in the [SimpleBaseline](https://arxiv.org/pdf/1804.06208.pdf) model.


### Dataset

In this project, we used [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid) as our dataset as it has 18 differect hand gestures with considerable variation in lighting, including artificial and natural light. Among those 18 different hand gestures, we selected 12 classes for our task. We further selected 2000 images in each classes for training and 200 images in each classes for testing in order to reduce the space of the dataset (the full data size is 716GB). Hand regions are extracted by cropping original images based on ground truth bounding box before training and testing. 

<img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/images/dataset.png" alt="dataset" height="200"/>

12 differect gestures are shown above.


### Model

<img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/images/model.png" alt="model" height="400"/>

Comparing with original PoseResNet, our model have an additional 1x1 convolutional layer after each deconvolutional layers and the predicted heatmap. We concatenate those intermediate features from additional convolutional layers with features extracted from backbone ResNet before using linear layers to output final results. We use L2 norm loss for landmark heatmap prediction and cross entropy loss for classification.

### Results

<img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/images/accuracy.png" alt="accuracy" height="300"/><img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/images/f1score.png" alt="f1score" height="300"/>

The above results indicate that our PoseResNet, utilizing a ResNet18 backbone, performs comparably to ResNext101. Notably, PoseResNet18 achieves this level of performance with 80% fewer parameters than ResNext101, highlighting the effectiveness of PoseResNet.


### Usage

1. **Dataset** </br>
Download images and annotations from [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid).
Afterwards, move annotation files to each image file which has the same class as the annotation file. Several sample images are provided in the [data/hagrid](https://github.com/kunnnnethan/hand-gesture-recognition/tree/v1/data/hagrid) file.
Your dataset file should look like the following:
    ```
    hagrid/
        ├── train_val_call/
            ├── ...jpg
            └── call.json
        ├── train_val_dislike/
            ├── ...jpg
            └── dislike.json
        ├── train_val_fist/
            ├── ...jpg
            └── fist.json
        ...
    ```
    (Optional) Run display_data.py to check if data are loaded correctly.
    ```
    python display_data.py
    ```
    (Optional) Run read_write_data.py if you want to create data with less number of images.
    ```
    python read_write_data.py
    ```

2. **Train** </br>
Modified arguments in [configs/train.yaml](https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/configs/train.yaml) file before training. Several augmentation methods are provided as well. Set the following arguments to True if augmentations are needed.
    ```yaml
    preprocess:
        rotate: False
        horizontal_flip: False
        hsv: False
    ```
    Afterwards, simply run train.py
    ```
    python train.py
    ```

3. **Test** </br>
Similarly, modified arguments in [configs/test.yaml](https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/configs/test.yaml) file before testing. Set the following argument to True if you want to visualize predicted result.
    ```yaml
    display_results: False
    ```
    Afterwards, run test.py
    ```
    python test.py
    ```
    You can also download weights that I trained for our project:
    * [poseresnet](https://drive.google.com/drive/folders/13_ikJTnck9HxytmD_eYnubQ3WxTMir9X?usp=share_link)
    
4. **Detect** </br>
    For inference, we use [SSDLite](https://sc.link/YXg2), which is provided by [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid) again, for cropping hand region from whole image. After extracting hand regions from images, it will be classified by our PoseResNet model.
    Again, modified arguments in [configs/detect.yaml](https://github.com/kunnnnethan/hand-gesture-recognition/blob/v1/configs/detect.yaml) file before inferencing.</br>
    Noted that ```img_size_for_detection: 320``` should be fixed for SSDLite hand detector; unless you have re-trained it.</br>
    Afterwards, run detect.py
    ```
    python detect.py
    ```

### References

[stefanopini/simple-HRNet](https://github.com/stefanopini/simple-HRNet)

**HaGRID - HAnd Gesture Recognition Image Dataset**
```
@article{hagrid,
    title={HaGRID - HAnd Gesture Recognition Image Dataset},
    author={Kapitanov, Alexander and Makhlyarchuk, Andrey and Kvanchiani, Karina},
    journal={arXiv preprint arXiv:2206.08219},
    year={2022}
}
```

**Simple Baselines for Human Pose Estimation and Tracking**
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
