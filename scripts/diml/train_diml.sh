dataset=${1:-cars196}
bs=${2:-112}
loss=${3:-margin_diml}
epochs=${4:-100}
seed=${5:-0}
datapath=/home/czhang/Drive2/datasets/revisit_dml
CUDA_VISIBLE_DEVICES=1,2 python train_diml.py --dataset $dataset --source_path $datapath --loss $loss --batch_mining distance --group ${dataset}_${loss} --seed $seed \
              --bs 112 --data_sampler class_random --samples_per_class 2 \
              --arch resnet50_diml_frozen_normalize  --n_epochs $epochs \
              --lr 0.00001 --embed_dim 128 --evaluate_on_gpu --evalevery 10
