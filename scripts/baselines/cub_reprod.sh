# """============= Baseline Runs --- cub200 ===================="""
main=train_baseline
dataset=cub200
datapath=/home/czhang/Drive2/datasets/revisit_dml

# resnet baseline
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize

# vit baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 96 --samples_per_class 2 --loss margin --batch_mining distance --arch vit_normalize --embed_dim 128

# swin t baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch swin_tiny_normalize --embed_dim 128

# cvt 13 baseline
CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 64 --samples_per_class 2 --loss margin --batch_mining distance --arch cvt_13_normalize --embed_dim 128

# deit small  baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b12_Distance_vit_small --loss_margin_beta 1.2 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch vit_small_normalize --embed_dim 128 --evalevery 10 --max_patience 100
