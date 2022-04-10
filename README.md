# noise-transition-matrix
COMP7250 course project of HKBU   
Reproducing VolMinNet to estimating the noise transition matrix   

##research paper
[Provably End-to-end Label-noise Learning without Anchor Points](https://arxiv.org/abs/2102.02400)


## Training
Example command to train the VolMinNet model in specified dataset
```bash
python train.py --dataset "mnist"   --flip_type "symmetric"  --noise_rate 0.2 --device  0
                          "cifar10"             "asymmetric"                           "cpu"
                          "cifar100"            "pair"                                
                                                                         
                                                                    
```
The quick command to run the experiments shown in papper
```commandline
sh benchmark.sh
```
## requirements
- torch
- torchvison
- numpy
- pandas
- tqdm

## Visualization
The code of visualization stores in Utils directory

## Dataset
1. MNIST
2. CIFAR10
3. CIFAR100

## Loss
- Cross Entropy
- Absolute log of Determinant of Transition Matrix
- Matrix estimation error

## Synthetic Noise Dataset
### Flip
1. Symmetry flipping
2. Pair flipping
3. Asymmetry flipping (Support)


## Hyper-Parameter
### MNIST
Model : LENet-5
 
epoch : 60  
batch size : 128  
momentum : 0.9  
weight decay : 1e-4  
learaning rate : 1e-2  

SGD -> h_\theta  
Adam -> trainsition matrix T  


### CIFAR10
Model : ResNet-18  

epoch: 150  
batch size : 128  
momentum : 0.9  
weight decay : 1e-4  
learaning rate : 1e-2  

SGD -> h_\theta and trainsition matrix T  
milestone [30, 60]  
gamma : 0.1

### CIFAR100
Model : ResNet-32 
 
epoch: 150  
batch size : 128  
momentum : 0.9  
weight decay : 1e-4  
learaning rate : 1e-2  

SGD  -> h_\theta  
Adam  -> trainsition matrix T     
milestone [30, 60]  
gamma : 0.1

## TODO
remove anchor points

## Reference
https://github.com/xuefeng-li1/Provably-end-to-end-label-noise-learning-without-anchor-points