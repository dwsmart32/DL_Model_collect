# Deep Learning Model Collect

## DL_tutorial

2021 Fall semester Seoul National University DL homework

- Making model tutorial

## DL_Unet

2021 Fall semester Seoul National University DL homework

- Basical Unet Model 

- Upgrade of basical model

## Plant_Seeding
2022 Spring Semester Peking University CVLD homework
- Dataset and Description below <br/> <br/> 
![스크린샷 2022-05-22 오전 12 42 59](https://user-images.githubusercontent.com/70640776/169661190-fc62e0f0-ae14-424b-a5bb-9ad5eb929fcb.png)

- [ kaggle ] (https://www.kaggle.com/competitions/plant-seedlings-classification)
#### 1. Traditional Method
##### How to run
```
python3 main.py --feature HOG --kernel rbf
```
```
[Model]
1. SIFT
2. HOG

[Classifier]
1. SVM(Linear, Gaussian)
2. K-means clustering
```
#### 2. Deep Learning Method
##### How to run
```
python3 main.py --net unet
```
```
[Model]
1. Resnet50
2. VGG16
3. SimpleNet
4. Unet

[Optimizer]
1. Adam
2. SGD
3. Adamgrad

[Augmentation]
1. RandomCrop
2. RandomFlip
3. Normalization

[Regularization]
1. Weight decay
2. Dropout
```

#### 3. [Result]
<br/>
<img width="1068" alt="스크린샷 2022-05-22 오전 12 59 03" src="https://user-images.githubusercontent.com/70640776/169661665-7049eabe-a598-40a0-9ddb-2c4e338a28f3.png">

```
train accuracy : 97.03%
valid accuracy : 79.26%
kaggle test accuracy : 81.17%
```

