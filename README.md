# noise-transition-matrix-
COMP7250 course project of hkbu —— reproduce estimating the noise transition matrix 


## metric
- Top-1 error
- Top-5 error
- Matrix estimation error
- 
## dataset
### noise
1. Symmetry flipping
2. Pair flipping

### remove anchor points (optional)

## Parameter
### MNIST
LENet-5

SGD -> h_\theta
batch size : 128
momentum : 0.9
weight decay : 10e-3
learaning rate : 10e-2

Adam -> matrix T
epoch: 60

### CIFAR10
ResNet-18

SGD -> $h_\theta$  T
batch size : 128
momentum : 0.9
weight decay : 10e3
learaning rate : 10e2

epoch: 150
divided by 10 after the 30th and 60th epoch

### CIFAR100
ResNet-32 

SGD  -> h_\theta  
batch size : 128
momentum : 0.9
weight decay : 10e3
learaning rate : 10e2

Adam  -> T
epoch: 150
divided by 10 after the 30th and 60th epoch


## requirements
- torch
- torchvison
- numpy