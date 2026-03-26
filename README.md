# YOLOv1 Implementation in PyTorch

From-scratch PyTorch implementation of YOLOv1 based on Redmon et al. (2015).

Paper: https://www.alphaxiv.org/abs/1506.02640

---

## How YOLO Works

Before YOLO, object detection was treated as a classification problem —
thousands of patches would be cropped from an image and classified one 
by one. YOLO reframes it as a regression problem, passing the image 
through a single network once to directly predict bounding box 
coordinates and class probabilities.

---

## Architecture

- 24 convolutional layers + 2 fully connected layers
- Input: 448x448
- Output: 7x7x30 tensor
    - 7x7 grid over the image
    - 30 values per cell = 2 boxes x 5 (x, y, w, h, confidence) + 20 classes


---

## Reference

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2015).
You Only Look Once: Unified, Real-Time Object Detection.
https://www.alphaxiv.org/abs/1506.02640
