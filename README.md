# Graduation Project 
# Multi-label classification performance improvement using data augmentation on deep learning
> ## 개요
> 본 프로젝트는 전기전자공학부 졸업논문으로 진행한 연구로서, 딥러닝 분야의 multi-label image classification 과제에 training data가 제한적인 경우에 있어서, accuracy를 증가시키기 위한 기법 중 하나인 Data Augmentation에 관한 연구입니다. 기존의 제한된 image data를 처리해 변형시킨 data를 생성하고 학습에 사용하여 accuracy를 보다 더 향상시킬 수있는 methods of augmentation, usages of augmentation 등을 실험 및 비교한 프로젝트입니다.   

# 개발환경   
language - Python3   
lib - TensorFlow   
OS - Linux, ubuntu   


# Task
20개의 Label(class)로 분류할수 있는 image data의 Classfication (한 image는 여러개 Label을 포함할 수 있습니다.)     
Image sets : 오픈데이터인 [VOC challenge 2007, 2012 sets](https://github.com/jonitox/GraduationProject/tree/main/voc/inputs)을 일정한 128x128x1 크기로 reshape하여 사용.   

# Model
Task를 수행하기 위한 CNN model을 설계하였습니다. model의 구성은 다음과 같습니다.   
* 3개의 Convolutional layers와 MAX-pooling layers. (3*3 filters, 2-zero-paddings, 1-stride 사용).    
* 이후 3개의 Fully-Connected layers (input으로 Convolutional layer output을 flatten)   
* Activation Function : ReLU  
* Cost: MSE(Min Squared Error) loss   
* 마지막 layer의 output을 Sigmoid 처리 후 One-hot encoding(0.5 기준)하여 최종 output출력.   

   
<img src="/description/model.png" width="600px" height="350px" alt="1"></img>   

# Applying Augmentation   
기존 image를 Flip, Drop out, Blur 등의 16가지 방식으로 처리해 data 생성.   
각 augmentation을 세가지 방식으로 적용해 학습하여 성능을 비교하였습니다.   
1. 16가지의 각 tranforamtion 중 한가지만 적용하여 생성한 data를 기존 이미지set과 같이 학습
2. 한 image에 몇가지 transformation을 동시에 적용하여 생성한 data를 기존 이미지set과 학습
3. 한 image에 몇가지 transformation을 각각 적용하여 생성한 data들 전부를 기존 이미지set과 학습   
   
<img src="/description/augmentation.png" width="350px" height="350px" alt="1"></img>   

# Result and Conclusion   
각 method로 모델을 학습시킨 후 test data로 accuracy를 측정하였습니다. augmented data를 이용한 경우 non-augmented test 대비 1~10%의 accuracy 상승을 보였습니다. 그중 가장 큰 상승은 single flip-augmentattion method에서 발생했으며, 약 8% 더높은 정확도를 보였습니다. 또한, 여러개의 method를 동시에 적용하거나, 여러번 적용시켜 augment한 Model이 single-augmentation methods에 비해 더 높은 정확도를 보이진 않았습니다. 이는 한 이미지의 간단한 augmentation이 현저한 information 변화를 가져오진 않으며, 과도하게 변형된 image는 machine에 불필요한 학습정보를 제공하여 Over-fitting과 같은 문제점을 야기할수도 있기 때문으로 여겨집니다.     

# File description
>- /(Report) Multi-label..rtf : [최종 Report File](https://github.com/jonitox/GraduationProject/blob/main/(Report)%20Multi-label%20classification%20performance%20improvement%20using%20data%20augmentation%20on%20deep%20learning.rtf)
>- /voc/inputs : [(original)VOC set](https://github.com/jonitox/GraduationProject/tree/main/voc/inputs)
>- /deep.py : [Python file for model](https://github.com/jonitox/GraduationProject/blob/main/deep.py)
