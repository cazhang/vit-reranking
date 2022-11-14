# """============= Baseline Runs --- MSLS ===================="""
main=train_msls_baseline
dataset=MSLS
datapath=/home/czhang/Drive2/datasets/revisit_dml

				
			
# distill cvt from vlad (RKD only)			
CUDA_VISIBLE_DEVICES=1,2 python train_msls_KDembed_tri.py --dataset $dataset --kernels 8 --source_path $datapath \
						--n_epochs 10 --group mini_MSLS_CVT128_VLAD128_triplet+RKD_distance_1e-4 --seed 0 --bs 6 --pool patchnetvlad \
						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
						--append_pca_layer --num_pcs 128 --save_path Training_Results \
						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
						--evalevery 1 --task_loss triplet --distill RKD --kd_weight 1.0 --tl_weight 1.0 --mini_data \
						--rkd_distance_weight 1.0 --rkd_angle_weight 0.0
						
