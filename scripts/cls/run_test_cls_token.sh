dataset=${1:-cub200}
embed_dim=${2:-128}
arch=${3:-cvt_13_normalize}
datapath=/home/czhang/Drive2/datasets/revisit_dml

CUDA_VISIBLE_DEVICES=0 python test_cls_token.py --dataset $dataset \
              --source_path $datapath \
              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
              --arch $arch --group cluster_cvt \
              --embed_dim $embed_dim --evaluate_on_gpu --not_pretrained