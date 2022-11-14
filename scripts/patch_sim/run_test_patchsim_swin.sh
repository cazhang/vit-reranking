dataset=${1:-cub200}
embed_dim=${2:-128}
arch=${3:-swin_tiny_normalize}
datapath=/home/czhang/Drive2/datasets/revisit_dml

CUDA_VISIBLE_DEVICES=0 python test_pair_patchsim_swin.py --dataset $dataset \
              --source_path $datapath \
              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
              --arch $arch --group patchsim_swin \
              --embed_dim $embed_dim --evaluate_on_gpu --to_submit