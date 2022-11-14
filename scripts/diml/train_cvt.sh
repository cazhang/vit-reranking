bs=${1:-112}
loss=${2:-margin_diml}
epochs=${3:-150}
seed=${4:-0}
dataset=${5:-cars196}
datapath=/home/czhang/Drive2/datasets/revisit_dml

# online_products

#CUDA_VISIBLE_DEVICES=1 python train_diml.py --dataset cub200 --source_path $datapath --loss $loss --batch_mining distance \
#              --group ${dataset}_$loss --seed $seed \
#              --bs $bs --data_sampler class_random --samples_per_class 2 \
#              --arch cvt_diml_normalize  --n_epochs $epochs \
#              --lr 0.00001 --embed_dim 128 --evaluate_on_gpu --group CVT_DIML_CLASS_T0.1_VU \
#              --use_cls_token --use_inverse --temperature 0.1

CUDA_VISIBLE_DEVICES=1,2 python train_diml.py --dataset $dataset --source_path $datapath --loss $loss --batch_mining distance \
              --group ${dataset}_$loss --seed $seed \
              --bs $bs --data_sampler class_random --samples_per_class 2 \
              --arch cvt_diml_normalize_frozen  --n_epochs $epochs \
              --embed_dim 128 --evaluate_on_gpu --group CVT_DIML_CLASS_Minus_clsNorm_frozen$bs \
              --use_cls_token --temperature 0.1 --evalevery 10 \
              --max_patience 5 --use_minus --use_inverse

#CUDA_VISIBLE_DEVICES=1,2 python train_diml.py --dataset online_products --source_path $datapath --loss $loss --batch_mining distance \
#              --group ${dataset}_$loss --seed $seed \
#              --bs 96 --data_sampler class_random --samples_per_class 2 \
#              --arch cvt_diml_normalize  --n_epochs 100 \
#              --lr 0.00001 --embed_dim 128 --evaluate_on_gpu --group CVT_DIML_CLASS_T0.1_VU \
#              --use_cls_token --use_inverse --temperature 0.1 --evalevery 5 \
#              --max_patience 3 --kernels 8
