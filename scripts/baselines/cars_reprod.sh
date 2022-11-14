# """============= Baseline Runs --- CARS196 ===================="""
main=train_baseline
dataset=cars196
datapath=/home/czhang/Drive2/datasets/revisit_dml

# resnet baseline
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize

#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance_res50_nofrozen --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_normalize

# vit baseline frozen
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch vit_frozen_normalize

# vit baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 96 --samples_per_class 2 --loss margin --batch_mining distance --arch vit_normalize --embed_dim 512

#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance_swinti --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch swin_tiny_normalize --embed_dim 128

# crossvit scratch
#CUDA_VISIBLE_DEVICES=1 python train_crossvit.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_Cross_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 48 --samples_per_class 2 --loss margin_cross --batch_mining distance --arch crossone_normalize --embed_dim 128 --cross_attn_depth 1 --skip_last_vit_norm --cat_global

# crossvit finetune
#CUDA_VISIBLE_DEVICES=2 python train_crossvit.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_Cross_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 48 --samples_per_class 2 --loss margin_cross --batch_mining distance --arch crossone_normalize_frozen --embed_dim 128 --cross_attn_depth 3 --skip_last_vit_norm --use_pretrained

# netvlad baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance_pvlad --loss_margin_beta 0.6 --seed 0 --bs 56 --samples_per_class 2 --loss margin --batch_mining distance --arch pvlad_normalize --embed_dim 128

# cvt 13 baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance_cvt --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch cvt_13_normalize --embed_dim 128

# cvt 13 baseline frozen
CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 6 --source $datapath --n_epochs 150 --group CARS_Margin_b06_Distance_cvt_trumc --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch cvt_13_normalize --embed_dim 128 --evalevery 10 --max_patience 5





