dataset=${1:-cars196}
embed_dim=${2:-128}
arch=${3:-cvt_13_normalize}
datapath=/home/czhang/Drive2/datasets/revisit_dml

# cub200
# cars196
# online_products


## cvt_13_normalize
CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset cub200 --source_path $datapath --seed 0 --bs 16 \
--data_sampler class_random --samples_per_class 2 --arch cvt_13_normalize --group diml_test --embed_dim $embed_dim --evaluate_on_gpu --use_inverse --use_cls_token --temperature 0.1 \
 --use_ot --grid_size 7 --plot_topk 2 --ot_part 1.0 --use_rollout

# diml_cvt_13_normalize training is True
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset cub200 --source_path $datapath --seed 0 --bs 16 \
#--data_sampler class_random --samples_per_class 2 --arch cvt_diml_normalize  --group diml_test --embed_dim $embed_dim --evaluate_on_gpu --use_inverse --use_cls_token --use_ot --temperature 0.1 --training

# cvt_13_normalize_frozen
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products --source_path $datapath \
#        --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 --grid_size 7\
#        --arch cvt_13_normalize_frozen  --group diml_test --embed_dim $embed_dim \
#        --evaluate_on_gpu  --use_cls_token --use_ot --temperature 0.1 --use_rollout

# cvt_13_normalize, rollout
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products \
#              --source_path $datapath \
#              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
#              --arch cvt_13_normalize_frozen --group diml_test_cvt \
#              --embed_dim $embed_dim --evaluate_on_gpu --grid_size 7 --temperature 0.1 \
#              --use_ot --ot_part 1.0 --use_minus --use_cls_token --to_submit --plot_topk 5
#
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products \
#              --source_path $datapath \
#              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
#              --arch cvt_13_normalize_frozen --group diml_test_cvt \
#              --embed_dim $embed_dim --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 \
#              --use_ot --ot_part 0.3 --use_minus --use_cls_token --use_rollout &
#
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products \
#              --source_path $datapath \
#              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
#              --arch cvt_13_normalize_frozen --group diml_test_cvt \
#              --embed_dim $embed_dim --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 \
#              --use_ot --ot_part 0.5 --use_minus --use_cls_token --use_rollout &
#
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products \
#              --source_path $datapath \
#              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
#              --arch cvt_13_normalize_frozen --group diml_test_cvt \
#              --embed_dim $embed_dim --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 \
#              --use_ot --ot_part 0.7 --use_minus --use_cls_token --use_rollout &
#
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products \
#              --source_path $datapath \
#              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
#              --arch cvt_13_normalize_frozen --group diml_test_cvt \
#              --embed_dim $embed_dim --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 \
#              --use_ot --ot_part 0.9 --use_minus --use_cls_token --use_rollout



#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products --source_path $datapath --seed 0 --bs 16 \
#--data_sampler class_random --samples_per_class 2 --arch cvt_13_normalize_frozen --group diml_test --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 --use_ot --use_cls_token --to_submit --use_minus --ot_part 0.7 &
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products --source_path $datapath --seed 0 --bs 16 \
#--data_sampler class_random --samples_per_class 2 --arch cvt_13_normalize_frozen --group diml_test --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 --use_ot --use_cls_token --to_submit --use_minus --ot_part 0.5 &
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products --source_path $datapath --seed 0 --bs 16 \
#--data_sampler class_random --samples_per_class 2 --arch cvt_13_normalize_frozen --group diml_test --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 --use_ot --use_cls_token --to_submit --use_minus --ot_part 0.3 &
#CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset online_products --source_path $datapath --seed 0 --bs 16 \
#--data_sampler class_random --samples_per_class 2 --arch cvt_13_normalize_frozen --group diml_test --evaluate_on_gpu --grid_size 7 --use_inverse --temperature 0.1 --use_ot --use_cls_token --to_submit --use_minus --ot_part 0.1
