# Fisher Task Distance and Its Applications in Transfer Learning and Neural Architecture Search
This is the source code for Fisher Task Distance and Its Applications in Transfer Learning and Neural Architecture Search paper (https://arxiv.org/abs/2103.12827).

## Description

We formulate an asymmetric (or non-commutative) distance between tasks based on Fisher Information Matrices. We provide proof of consistency for our distance through theorems and experiments on various classification tasks. We then apply our proposed measure of task distance in transfer learning
on visual tasks in the Taskonomy dataset. Additionally, we show how the proposed distance between a target task and a set of baseline tasks can be used to reduce the neural architecture search space for the target task. The complexity reduction in search space for task-specific architectures is achieved by building on the optimized architectures for similar tasks instead of doing a full search without using this side information. Experimental results demonstrate the efficacy of the proposed approach and its improvements over other methods.

## Getting Started

### Dependencies

* Requires Pytorch, Numpy
* MNIST dataset (https://www.kaggle.com/oddrationale/mnist-in-csv)
* CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
* CIFAR-100 (https://www.cs.toronto.edu/~kriz/cifar.html)
* ImageNet (https://image-net.org/challenges/LSVRC/index.php)
* Taskonomy (http://taskonomy.stanford.edu/)

### Executing program

* First, we define tasks in MNIST, CIFAR-10, CIFAR-100, ImageNet, Taskonomy datasets and use the CNN to train on each task. The weights of the trained CNN is saved for each task.
```
python train_task_mnist.py
python train_task_cifar.py
python train_task_cifar100.py
python train_task_taskonomy.py
```
* Next, we compute the Fisher Information matrices for each pair of tasks using the base task's network. Then, we identify the closest tasks based on the Fr\'echet of the Fisher Information matrices
```
python fisher-distance.py
python fisher-distance_taskonomy.py
```
Lastly, the FUSE algorithm is applied to find the suitable architecture for the incoming task:
```
python NAS_FUSE.py
```

### Results
The confusion matrices below shows the mean (left) and standard deviation (right) of the distances between 8 baseline tasks from MNIST, CIFAR-10 datasets.
<p align="center">
  <img src="images/fig1.jpg" height="350" title="Mean">
  <img src="images/fig2.jpg" height="350" title="Sig">
</p>

The table below indicates the comparison of the NAS performance with handdesigned classifiers and state-of-the-art methods on Task 3 in
MNIST based on the discovered closest task, Task 7 
| Architecture | Accuracy (%) | Paramameters (M) | GPU days |
| :---         |    :---:  |     :---:        |  :---:   |
| VGG-16       | 99.55     |  14.72    | - |
| ResNet-18    | 99.56     |  11.44    | - |
| DenseNet-121 | 99.61     |  6.95     | - |
| Random Search| 99.59     |  2.23     | 4 |
| ENAS         | 97.77     |  4.60     | 4 |
| DARTS        | 99.51     |  2.37     | 2 |
| LD-NAS (ours)| 99.67     |  2.28     | 2 |

The table below indicates the comparison of the NAS performance with handdesigned classifiers and state-of-the-art methods on Task 6 in
CIFAR-10 based on the discovered closest task, Task 7. 
| Architecture | Accuracy (%) | Paramameters (M) | GPU days |
| :---         |    :---:  |     :---:        |  :---:   |
| VGG-16       | 86.75     |  14.72    | - |
| ResNet-18    | 86.93     |  11.44    | - |
| DenseNet-121 | 88.12     |  6.95     | - |
| Random Search| 88.55     |  3.65     | 5 |
| ENAS         | 75.22     |  4.60     | 4 |
| DARTS        | 90.11     |  3.12     | 2 |
| LD-NAS (ours)| 90.87     |  3.02     | 2 |

## Authors

Cat P. Le (cat.le@duke.edu), 
<br>Mohammadreza Soltani, 
<br>Juncheng Dong, 
<br>Vahid Tarokh