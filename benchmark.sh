#!/bin/bash


for dataset in "mnist" "cifar10" "cifar100"
do
    for noise_rate in 0.2 0.5
    do
        python3 train.py --dataset $dataset  --flip_type  'symmetric' --noise_rate $noise_rate --device 0
    done
done

for dataset in "mnist" "cifar10" "cifar100"
do
    for noise_rate in 0.2 0.45
    do
        python3 train.py --dataset $dataset  --flip_type  'pair' --noise_rate $noise_rate --device 0
    done
done

for dataset in "mnist" "cifar10" "cifar100"
do
    for noise_rate in 0.2 0.5
    do
        python3 train.py --dataset $dataset  --flip_type  'symmetric' --noise_rate $noise_rate --device 0 --without_vol
    done
done

for dataset in "mnist" "cifar10" "cifar100"
do
    for noise_rate in 0.2 0.45
    do
        python3 train.py --dataset $dataset  --flip_type  'pair' --noise_rate $noise_rate --device 0 --without_vol
    done
done



