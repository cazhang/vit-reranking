# """============= Baseline Runs --- MSLS ===================="""
main=train_msls_baseline
dataset=MSLS
datapath=/home/czhang/Drive2/datasets/revisit_dml

					
# cvt13, new loss ada + KD, no xbm		
# CUDA_VISIBLE_DEVICES=0 python train_msls_KDembed_xbm.py --dataset $dataset --kernels 4 --source_path $datapath \
						# --n_epochs 30 --group miniMSLS_CVT_128_VLAD128_adaloss+kd20 --seed 0 --bs 3 --pool patchnetvlad \
						# --arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
						# --vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
						# --append_pca_layer --num_pcs 128 --save_path Training_Results \
						# --imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
						# --evalevery 1 --task_loss xbm_triplet --distill embed --kd_weight 0.0 --tl_weight 1.0 --xbm_weight 1.0 \
						# --rkd_distance_weight 1.0 --rkd_angle_weight 1.0 --mini_data --xbm_start_iteration 3 --xbm_size 10000 --debug
						
# cvt13, new loss ada+ KD, with xbm		
CUDA_VISIBLE_DEVICES=1,2 python train_msls_KDembed_xbm.py --dataset $dataset --kernels 4 --source_path $datapath \
						--n_epochs 30 --group miniMSLS_CVT_128_VLAD128_adaSimloss+kd100 --seed 0 --bs 6 --pool patchnetvlad \
						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
						--append_pca_layer --num_pcs 128 --save_path Training_Results \
						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
						--evalevery 1 --task_loss xbm_triplet --distill embed --kd_weight 100.0 --tl_weight 1.0 --xbm_weight 1.0 \
						--rkd_distance_weight 1.0 --rkd_angle_weight 1.0 --mini_data --xbm_start_iteration 0 --xbm_size 8000
						
