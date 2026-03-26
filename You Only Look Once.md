Tags : #deeplearning , #cv , #cnn , #research_paper
Link : https://www.alphaxiv.org/abs/1506.02640



we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. 
Before YOLO AI models treated finding objects as a ***classification*** problem. They would crop thousands of tiny patches from an image and ask the AI . "is there a dog in this patch?"
YOLO flipped the script by turning it into a ***regression*** problem. Instead of asking yes or no questions about the image patches, YOLO asks the neural network to calculate the exact numbers that represent where the object is.

## Introduction 
- most recent approaches use region proposal methods to first generate potential bounding boxes in an image and then run a classifier on these proposed boxes. after classification, post processing is used to refine the bounding  boxes, eliminate duplicate detections and rescore boxes based on other objects in the scene.
- these complex pipelines are slow and hard to optimize because each individual component must be trained separately.
- we reframe object detection as a single regression problem, straight from image pixels to bounding box coordinated and class probabilities.
- using our system ***you only look once*** at an image to predict what objects are present and where they are.

## The YOLO detection system
- a single convolutional neural network simultaneously predicts multiple bounding boxes and class probabilities for those boxes.
- YOLO trains on full images and directly optimizes detection performance.
- YOLO is extremely fast. We treat frame detection as a regression problem we don't need a complex pipeline.
- we simply run a neural network on a new image at test time to predict detections. our base network runs at `45 frames per second` and a fast version runs at `150 frames per second`. 
- YOLO sees the entire image during training as well as test time so it implicitly encodes contextual information about classes as well as their appearance.
- YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, YOLO outperforms top detection methods.

## Unified Detection
- we unify the separate components of object detection into a single neural network. the network uses features from the entire image to predict each bounding box.
- our network reasons globally about the full image and all the objects in the image
- Our system divides the input image into an ***`S × S grid`***. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object
- Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts.
- Each bounding box consists of 5 predictions: `x, y, w, h, and confidence`.
- For evaluating YOLO on PASCAL VOC, we use` S = 7 (grid size), B = 2(boxes per cell) `. PASCAL `VOC has 20 labelled classes so C = 20`. ***Our final prediction is a` 7 × 7 × 30 tensor`.***
- The tensor shape comes from each grid cell predicting B bounding boxes (5 values each: x, y, w, h, confidence) + C class probabilities. So ***`7 × 7 × (2×5 + 20) = 7 × 7 × 30`***
- After predictions are made, ***Non-Maximum Suppression (NMS)*** is applied to eliminate duplicate detections of the same object , the box with the highest confidence score is kept, others overlapping it heavily are discarded.
### Network design 
- The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates. 
- The network has ***24 convolutional layers*** followed by ***2 fully connected layers***.
- We simply use `1 x 1` reduction layers followed by `3 x 3`  convolutional layers. 
- The final output of our network is a `7 x 7 x 30` tensor of predictions.
- The architecture follows this sequence: Conv layers (feature extraction) → FC layers (prediction). ***Input image is*** ***`448 × 448 × 3`***, and the final ***output is the `7 × 7 × 30` tensor.

![[Pasted image 20260325223150.png]]
### Training 
- We achieve a single crop top-5 accuracy of 88% on the ImageNet 2012 validation set. 
- We add four convolutional layers and two fully connected layers with randomly initialized weights. 
- We increase the input resolution of the network from `224 x 224 to 448 x 448 `.
- Our final layer predicts both class probabilities and bounding box coordinates. 
- We use a ***linear activation function for the final layer***, and ***all other layers use the leaky rectified linear activation.***
- YOLO predicts multiple bounding boxes per grid cell.
- If a grid square is empty, YOLO does not penalize it for grading the wrong category of the object.
- We train the network for about ***135 epochs*** on training and validation sets. Throughout training we use a ***batch size of 64***, a ***momentum of 0.9***, and a ***decay of 0.0005***.
- Learning rate schedule is as follows :
	- For the first epochs we slowly raise the learning rate from 10−3 to 10−2
	- Continue training with the same learning rate for 75 epochs.
	- then 10−3 for 30 epochs, and finally 10−4 for 30 epochs
- Avoid overfitting, views dropout, and extensive data augmentation.
- A ***dropout layer with the rate 0.5 after the first connected layer*** prevents co-adaptation between layers.
- Introduce random scaling and translations of up to 20% on the original image size.
- We also randomly adjust the exposure and saturation of the image by up to a factor of 1.5.
### Limitations of YOLO
- The root cause of most limitations: each grid cell can only predict **2 bounding boxes** and **1 class**. So if two objects of different classes overlap in the same cell, one gets missed entirely.
- Struggles with crowded objects.
- It has difficulty in identifying unusual shapes.
- Imprecise boxes for small items

## Notes to self
- Bounding box is a rectangle drawn around an image. It has 4 numbers :
	- (x , y) -> center of the box
	- width
	- height