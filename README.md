# SpatialAtten
Spatial Attention-Based Deep Learning System for Breast Cancer Pathological Complete Response Prediction with Serial Histopathology Images in Multiple Stains


```
SpatialAtten
│   README.md
│   model.py            ## YOLOv4 model description    
|   trainDetector.py    ## train the YOLOv4 for tumor cells and TILs detection
|   trainPredictor.py   ## freezing detecor part and train the whole system with PCR prediction
│
└───class_names
│   │   classes.txt     ## class names for detection
|
└───dataset
│   └───img
│   |   │   img.jpg     ## all images for detection training
|   └───txt
│       │   anno.txt    ## all annotations for detection training
|
└───PCR
│   └───pcr
│   |   │   he.jpg      ## all HE images for PCR prediction from pcr cases
│   |   │   ki-67.jpg   ## all Ki-67 images for PCR prediction from pcr cases
│   |   │   phh3.jpg    ## all PHH3 images for PCR prediction from pcr cases
│   └───non-pcr
│   |   │   he.jpg      ## all HE images for PCR prediction from non-pcr cases
│   |   │   ki-67.jpg   ## all Ki-67 images for PCR prediction from non-pcr cases
│   |   │   phh3.jpg    ## all PHH3 images for PCR prediction from non-pcr cases

```

Firstly run the trainDetector.py to train the tumor cells and TILs detector. Then run trainPredictor.py to train the PCR predictor.

The YOLOv4 part is referred from https://github.com/taipingeric/yolo-v4-tf.keras.git

# Reference
@inproceedings{duanmu2021spatial,\
  title={Spatial Attention-Based Deep Learning System for Breast Cancer Pathological Complete Response Prediction with Serial Histopathology Images in Multiple Stains},\
  author={Duanmu, Hongyi and Bhattarai, Shristi and Li, Hongxiao and Cheng, Chia Cheng and Wang, Fusheng and Teodoro, George and Janssen, Emiel AM and Gogineni, Keerthi and Subhedar, Preeti and Aneja, Ritu and Jun Kong},\
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},\
  pages={550--560},\
  year={2021},\
  organization={Springer}\
}
