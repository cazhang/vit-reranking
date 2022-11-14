### Step 1: pre-train a base model using global image representation

CUDA_VISIBLE_DEVICES=0 python train_baseline.py --dataset cub200 \
            --kernels 6 --source $datapath --n_epochs 150 --group CUB_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch cvt_13_normalize --embed_dim 128

NOTE: use dataset {cub200, cars196, online_products} and set datapath accordingly

### Step 2: validate reranking using structural similarity

CUDA_VISIBLE_DEVICES=0 python test_diml_cvt.py --dataset cub200 --source_path $datapath \
            --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 --arch cvt_13_normalize --group test_cvt_cub200 --embed_dim 128 --evaluate_on_gpu --use_cls_token --temperature 0.1 --use_ot --use_inverse --grid_size 7 --plot_topk 5 --ot_part 1.0 --use_rollout

### (optional) Step 3: visualise patch similarity
CUDA_VISIBLE_DEVICES=0 python test_pair_patchsim_cvt.py --dataset cub200 \
              --source_path $datapath \
              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2 \
              --arch cvt_13_normalize --group patchsim_cvt_cub200 \
              --embed_dim 128 --evaluate_on_gpu --to_submit

The code is based on DIML: https://github.com/wl-zhao/DIML and 
RevisitDML: https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
