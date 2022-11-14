# """============= Baseline Runs --- cub200 ===================="""
main=train_baseline
dataset=online_products
datapath=/home/czhang/Drive2/datasets/revisit_dml

# resnet baseline
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group OP_Margin_b06_Distance_res50 --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize

# vit baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 96 --samples_per_class 2 --loss margin --batch_mining distance --arch vit_normalize --embed_dim 128

# swin t baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch swin_tiny_normalize --embed_dim 128

# cvt 13 baseline
CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 100 --group OP_Margin_b06_Distance_cvt_frozen1 --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch cvt_13_normalize_frozen --embed_dim 128 --evalevery 5 --max_patience 4

# cvt 13 baseline sgd
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 100 --group OP_Margin_b06_Distance_cvt_sgd_1e-4 --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch cvt_13_normalize --embed_dim 128 --optim sgd --tau 20 --gamma 0.5 --lr 0.0001

# cvt 13 fpn baseline
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 12 --samples_per_class 2 --loss margin --batch_mining distance --arch cvt_13_normalize_fpn --embed_dim 128

# crossvit scratch
#CUDA_VISIBLE_DEVICES=1 python train_crossvit.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_Cross_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 48 --samples_per_class 2 --loss margin_cross --batch_mining distance --arch crossone_normalize --embed_dim 128 --cross_attn_depth 1 --skip_last_vit_norm --cat_global

# crossvit finetune
#CUDA_VISIBLE_DEVICES=2 python train_crossvit.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_Cross_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 48 --samples_per_class 2 --loss margin_cross --batch_mining distance --arch crossone_normalize --embed_dim 128 --cross_attn_depth 1 --skip_last_vit_norm --cat_global --use_pretrained

