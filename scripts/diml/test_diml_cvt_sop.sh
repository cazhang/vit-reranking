dataset=${1:-cars196}
embed_dim=${2:-128}
arch=${3:-cvt_13_normalize}
datapath=/home/czhang/Drive2/datasets/revisit_dml



# cvt_13_normalize, rollout
CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products \
              --source_path $datapath \
              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
              --arch cvt_13_normalize_frozen --group diml_test_cvt \
              --embed_dim $embed_dim --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 \
              --use_ot --ot_part 1.0 --use_minus --use_cls_token