# hand-gesture-recognition

This model is utilized in the hand gesture recognition system within the MeCO robot at the Interactive Robotics and Vision Laboratory, University of Minnesota.

<img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/main/images/demo.gif" alt="demo" width="500"/>

### Introduction

<img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/main/images/model.png" alt="model" height="200"/>

The hand gesture recognition system consists of two main parts: hand detection and gesture classification. Initially, the YOLOv7-tiny algorithm detects the region where hands are present in the given image. After detection, these regions are cropped from the images and then classified using a multitasking network. The structure of this multitasking network is shown above.

Our multitasking network leverages enriched features to attain high-performance classification with fewer model parameters. We achieved this by training the network jointly on the classification and pose estimation tasks, with the pose estimation task serving as an auxiliary task that is not used in the application. The network starts by generating dense features from input images using a CNN backbone. These features are then combined with a learnable class embedding and fed into a ViT encoder. Afterward, the class embedding is separated from the features and classified by a linear layer. To further enhance the transformer's learned features, the remaining features are decoded by a simple decoder to generate hand poses.


### Dataset

We trained our model on the large-scale hand gesture dataset: [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid). We include the no_gesture class, resulting in 19 classes in total.

### Results

<img src="https://github.com/kunnnnethan/hand-gesture-recognition/blob/main/images/confusion_matrix.png" alt="confusion_matrix" height="700"/>

This result is tested on the HaGRID test data.


### Usage

1. **Environment** </br>

    Build Docker environment
    ```bash
    docker build -t hand-gesture:latest docker/
    ```
    
    Run container
    ```bash
    docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --network="host" \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -v <PATH_OF_THE_REPOSITORY>:/workspace \
        hand-gesture:latest
    ```

1. **Dataset** </br>
Download images and annotations from [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid).

    To reduce training time and save CPU resources, we crop the hand region first from the original data:
    ```bash
    python extract_data.py --root_dir <DOWNLOADED_HAGRID_DATA>
    ```
    (Optional) Run display_data.py to check if data are loaded correctly.
    ```bash
    python display_data.py
    ```

2. **Train** </br>
    ```bash
    python train.py \
        --data_config configs/hagrid.yaml \
        --suffix best \
        --batch_size 32 \
        --num_workers 8 \
        --epochs 40 \
        --lr 0.0001 \
        --lr_step 30 \
        --image_size 192 192 \
    ```

3. **Export** </br>
    Export PyTorch model to ONNX
    ```bash
    python export.py \
        --data_config configs/hagrid.yaml \
        --image_size 192 192
        --weight_path <YOUR_MODEL_WEIGHT_PATH>
    ```
    You can also download the [pretrained model](https://drive.google.com/file/d/1gtGPClNuARtZHsyX595p0VBBCqJDOqxV/view?usp=sharing).
    
4. **Inference** </br>
    For inference, we use YOLOv7-tiny to detect the hand region from the whole image. The [detector](https://drive.google.com/file/d/16HTdppn7gvbuPTLh7DZn01vbNU-E_Xvu/view?usp=sharing) is trained on data collected by the Interactive Robotics and Vision Laboratory at the University of Minnesota. After extracting hand regions from images, it will be classified by the multi-tasking model. The path of the inference data should be either a video file or a folder containing image files.
    ```bash
    python detect.py \
        --data_config configs/hagrid.yaml \
        --cls_weight gesture-classifier.onnx \
        --det_weight yolov7-tiny-diver.onnx \
        --data_path <YOUR_TEST_DATA>
    ```

### References

- *Y. Xu, J. Zhang, Q. Zhang, and D. Tao*, **Vitpose: Simple vision transformer baselines for human pose estimation**, arXiv preprint arXiv:2204.12484, 2022.
- *A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby*, **An image is worth 16x16 words: Transformers for image recognition at scale**, ICLR, 2021.
- *A. Kapitanov, K. Kvanchiani, A. Nagaev, R. Kraynov, and A. Makhliarchuk*, **HaGRID - HAnd Gesture Recognition Image Dataset**, in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 4572–4581, January 2024.
- *B. Xiao, H. Wu, and Y. Wei*, **Simple Baselines for Human Pose Estimation and Tracking**, in European Conference on Computer Vision (ECCV), 2018.
