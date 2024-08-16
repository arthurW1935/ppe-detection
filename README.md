# PPE Detection Model

## Introduction
The aim of this was to build a system to ensure the safety of the workers of any construction site or a factory by checking whether the workers are equipped with proper PPE (personal protective equipment) or not.

Detailed document [here](https://docs.google.com/document/d/1OgHGKbq-g3xVSKZ1191oWQBTiWhP-ADwTxhfVMx-X_s/edit?usp=sharing).

## Setup
Clone the repository
```
git clone https://github.com/arthurW1935/ppe-detection.git
cd ppe-detection
```

Create a virtual environment (You must have Python in your system)
```
python -m venv venv
venv\Scripts\activate
```

Now install all the dependencies using requirements.txt
```
pip install -r requirements.txt
```

## Running the application
To run the program, you can run the following in your terminal
```
python inference.py /path/to/input/images /path/to/output/images
```

## General Approach
The overall approach was to build two models - one to detect a person from the image, and second to detect PPE kits from the cropped images. We will feed the image first to the person detection model, and then crop the image accordingly, and check for each person using the second model. 

## Detailed Steps
- **Studying the dataset:** It was a dataset of 416 images with the labels in pascalVOC format, with 10 classes, 0 being the person and the rest 9 different ppe-kits.
- **Model Selection:** I selected YoloV8 for this project, as it is one of the famous state-of-the-art models for object detection. 
- **Annotation Conversion:** Since we decided to use YoloV8, we needed to convert the XML annotations in PascalVOC format to YoloV8 format. Wrote a simple Python Script for that.
- **Dataset Separation:** As we need two different models in this case, I separated the main dataset into two different datasets based on the use case. I cropped the original dataset images based on the person bounding box and created a separate dataset for the PPE detection model. 
     Also, change of annotations was also needed to acknowledge the shift of the image during cropping. Did that using this formula:
  ```
   crop_ppe_x1 = max(0, ppe_x1 - x1)
   crop_ppe_y1 = max(0, ppe_y1 - y1)
   crop_ppe_x2 = min(crop_width, ppe_x2 - x1)
   crop_ppe_y2 = min(crop_height, ppe_y2 - y1)
  ```
   After this separation, I split the dataset into train, test, and validation sets for training purposes.
- **Training the Model:** Trained both the YoloV8 model on the datasets, and achieved 0.6 box_loss in the Person Detection Model and 0.96 box_loss in the PPE Detection Model. Both the model was trained for 30 epochs.
- **Final Detection System:** This program takes the directory of images as an input and saves the images with the predicted PPE-kits in the desired output directory. 
