dataset=${1:-cub200}
embed_dim=${2:-128}
arch=${3:-vit_small_normalize}
datapath=/home/czhang/Drive2/datasets/revisit_dml

CUDA_VISIBLE_DEVICES=0 python test_diml_vit.py --dataset $dataset --source_path $datapath --seed 0 --bs 16 \
											   --data_sampler class_random --samples_per_class 2 \
											   --arch $arch --group diml_test_vit --embed_dim $embed_dim --evaluate_on_gpu \
											   --grid_size 7 --use_ot --ot_part 1.0 --use_cls_token --use_soft --to_submit