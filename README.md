# IMTA

This repository contains the code (in PyTorch) for "[Improved Techniques for Training Adaptive Deep Networks](https://arxiv.org/pdf/1908.06294.pdf)" paper by [Hao Li](https://github.com/cpsxhao)\*, [Hong Zhang](https://github.com/kalviny)\*, [Xiaojuan Qi](https://xjqi.github.io/), [Ruigang Yang](http://research.baidu.com/People/index-view?id=114) and [Gao Huang](http://www.gaohuang.net/) (* Authors contributed equally).

## Introduction

This paper presents three techniques to improve the training efficacy of adaptive deep networks from two aspects: (1) a Gradient Equilibrium algorithm to resolve the conflict of learning of different classifiers; (2) an Inline Subnetwork Collaboration approach and a One-for-all Knowledge Distillation algorithm to enhance the collaboration among classifiers. 

### Method Overview.

<img src="https://user-images.githubusercontent.com/799931/63514741-e2b7dc80-c51b-11e9-84e2-d95da7b92024.jpg" width="650">



## Results
### (a) Budgeted prediction results on ImageNet.

<img src="https://user-images.githubusercontent.com/799931/63515308-219a6200-c51d-11e9-956d-2e0cb12026fa.jpg" width="400">

### (b) Budgeted prediction results on CIFAR-100.

<img src="https://user-images.githubusercontent.com/799931/63515429-6de5a200-c51d-11e9-96b9-2bc9d63a1ae1.jpg" width="400">``


## Dependencies:

+ Python3
+ PyTorch >= 1.0

## Usage
We Provide shell scripts for training an MSDNet on ImageNet with GE, ISC and OFA.
### Training an IMTA_MSDNet (block=5, step=4) on ImageNet.
+ **Step 1: Training an MSDNet with GE from scratch**  
    Modify the run_GE.sh to config your path to the ImageNet, your GPU devices and your saving directory. Then run
    ```
    bash run_GE.sh
    ```

+ **Step 2: Training the classifiers with ISC and OFA**    
    Modify the run_IMTA.sh to config your path to the ImageNet, your GPU devices and your saving directory (different from the saving directory of your GE_MSDNet). Please be noted that the MSDNet settings should be exactly the same as your trained GE_MSDNet, and the pretrained directory of IMTA_MSDNet should be the saving directory of your trained GE_MSDNet.
    ```
    bash run_IMTA.sh
    ```

