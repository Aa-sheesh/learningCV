# **Learning Computer Vision**

## Roadmap

![roadmap](./static/roadmap.png)

# Resource 1 : Introduction to CV

Access the resource [here.](https://opencv.org/blog/what-is-computer-vision/)

## Working

### Step 1 : Feature Extraction

- The system scrutinizes the incoming visual data to identify and isolate significant visual elements, such as edges, shapes, textures, and patterns.
- These features are critical because they serve as the building blocks for the subsequent stages of analysis.
- To facilitate computer processing, these identified features are translated into numerical representations, effectively converting the visual information into a format that machines can comprehend and manipulate more efficiently.

### Step 2 : Object Detection

- System’s algorithms work to identify and locate specific objects or entities within the images.

### Step 3 : Image Classification

- Image classification involves categorizing entire images into predefined classes or categories. This is where Convolutional Neural Networks (CNNs) come into play.

### Step 4 : Object Tracking

- It involves the ability to monitor and trace the movement of objects as they traverse through consecutive frames of a video.

### Step 5 : Semantic Segmentation

- Labeling each and every pixel within an image with its respective category.

## Key Features of CV

1. Visual Perception
    - seeks to replicate the human ability to perceive and process visual information. It achieves this by capturing and comprehending images or video data from cameras and sensors.

2. Image Understanding
    - This process involves recognizing a wide array of elements, from objects and scenes to people, and understanding their attributes and relationships within the visual context.

3. Pattern Recognition
    - This encompasses the identification of shapes, textures, colors, and various intricate details that form the building blocks of our visual world.

4. Machine Learning and Deep Learning
    - At the core of Computer Vision lies machine learning and deep learning techniques.

5. Multidisciplinary Character
    - This amalgamation of insights from various domains enables the creation of systems capable of understanding and interpreting visual data with remarkable precision.

## Tasks

1. Image Classification
2. Object Detection
3. Image Segmentation
4. Facial Recognition
    - enhancing security through authentication and access control to adding fun filters in entertainment and aiding law enforcement in identifying suspects from surveillance footage.
5. Pose Estimation
    - used in fitness tracking, gesture recognition, and gaming
6. Scene Understanding
    - This capability is crucial in robotics, augmented reality, and smart cities for tasks like navigation, context-aware information overlay, and traffic management.
7. OCR - Optical Character Recognition
    - Applications range from document management to text translation and accessibility tools for visually impaired individuals.
8. Image Generation
    - GANs can create realistic images

## Company Use cases

1. Intel
    - End-to-End AI Pipeline Software
    - Intel Distribution
    - Intel Geti : an open-source, enterprise-class Computer Vision platform.
    - Hardware Portfolio for Diverse Needs : offer a broad hardware portfolio that provides the processing power needed for deploying Computer Vision in diverse environments
    - Open Source Tools for Scalability
2. Nvidia
    - NVIDIA Maxine : enhance audio and video quality in real-time, adding augmented reality effects
    -
3. Qualcomm
    - reshaping the landscape of Computer Vision in both consumer and enterprise IoT domains
4. Meta
    - platforms and products to create more immersive experiences and enhance user safety

# Resource 2 : Object Detection using yolo v8

Access link [here.](https://docs.ultralytics.com/tasks/detect/)

Github link [here.](https://github.com/ultralytics/ultralytics)

## 1. Train

- Train YOLO11n on the COCO8 dataset for 100 epochs at image size 640.

```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

- [Configuration List](https://docs.ultralytics.com/usage/cfg/)

## 2. Dataset formats

- [dataset guide](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
- for converting json to yolo format try [this](https://github.com/ultralytics/JSON2YOLO)

## 3. Val

for validation of YOLO11n on the COCO8 dataset.

```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n.pt")  # load an official model
    model = YOLO("path/to/best.pt")  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
```

## 4. Predict

Use a trained YOLO11n model to run predictions on images.

```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n.pt")  # load an official model
    model = YOLO("path/to/best.pt")  # load a custom model

    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
```

## 5. Export

Export a YOLO11n model to a different format like ONNX, CoreML, etc.

```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n.pt")  # load an official model
    model = YOLO("path/to/best.pt")  # load a custom trained model

    # Export the model
    model.export(format="onnx")
```

## FAQs

1. How do I train a YOLO11 model on my custom dataset?
    - Prepare the Dataset: Ensure your dataset is in the YOLO format. For guidance, refer to our Dataset Guide.
    - Load the Model: Use the Ultralytics YOLO library to load a pre-trained model or create a new model from a YAML file.
    - Train the Model: Execute the train method in Python or the yolo detect train command in CLI.

    ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n.pt")

        # Train the model on your custom dataset
        model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)
    ```

2. What pretrained models are available in YOLO11?
    - Some available models are:
        1. YOLO11n
        2. YOLO11s
        3. YOLO11m
        4. YOLO11l
        5. YOLO11x

    - All models [here](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

3. Why should I use Ultralytics YOLO11 for object detection?
    - Designed to offer state-of-the-art performance for object detection, segmentation, and pose estimation
        1. Pretrained Models: Utilize models pretrained on popular datasets like COCO and ImageNet for faster development.
        2. High Accuracy: Achieves impressive mAP scores, ensuring reliable object detection.
        3. Speed: Optimized for real-time inference, making it ideal for applications requiring swift processing.
        4. Flexibility: Export models to various formats like ONNX and TensorRT for deployment across multiple platforms.

## Projects Practise

1. Automatic Number Plate Recognition using yolo v12

### Step 1: Train Yolov8 object detection on a custom dataset

[Link](https://www.youtube.com/watch?v=m9fH9OWn8YM)

- Train Yolov8 object detection on a custom dataset

1. Data Annotation - use cvat to annotate data and export as yolo format
2. Train yolov11

# Resource 3 : NPTEL Course on Image Transformations and CV

Access the course [here.](https://onlinecourses.nptel.ac.in/noc21_ee23/preview)
