dataset=${1:-cub200}
embed_dim=${2:-128}
arch=${3:-resnet50_frozen_normalize}
datapath=/home/czhang/Drive2/datasets/revisit_dml

# test res50_diml sop
CUDA_VISIBLE_DEVICES=0 python test_diml_base.py --dataset $dataset \
              --source_path $datapath \
              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
              --arch resnet50_frozen_normalize --group triplet_res50 \
              --embed_dim $embed_dim --evaluate_on_gpu --grid_size 7  --use_cls_token \
              --plot_topk 5
