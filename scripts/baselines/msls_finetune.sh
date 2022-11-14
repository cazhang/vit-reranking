# """============= Baseline Runs --- MSLS ===================="""
main=train_msls_baseline
dataset=MSLS
datapath=/home/czhang/Drive2/datasets/revisit_dml

						
# distill cvt from vlad (pre-embeding + triplet )
#CUDA_VISIBLE_DEVICES=2 python train_msls_baseline.py --dataset $dataset --kernels 8 --source_path #$datapath --n_epochs 30 --group MSLS_CVT_finetune_triplet+KD_1e-4 --loss_margin_beta 0.6 --seed 0 --bs 4 #--pool patchnetvlad \
#						--samples_per_class 2 --loss margin --batch_mining distance \
#						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
#						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
#						--append_pca_layer --num_pcs 128 --save_path Training_Results \
#						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
#						--evalevery 1 \
#						--resume_path /home/czhang/DIML/Training_Results/MSLS/Apr08_15-57-57_MSLS_CVT_distill_embed_1e-4/checkpoints/model_best.pth.tar \
#						--distill embed --mini_data
						
						
# distill cvt from vlad (first embeding + triplet )
#CUDA_VISIBLE_DEVICES=1,2 python train_msls_KDembed_tri.py --dataset $dataset --kernels 8 --source_path $datapath --n_epochs 30 --group MSLS_CVT_finetune_triplet+KD10_1e-4 --loss_margin_beta 0.6 #--seed 0 --bs 6 --pool patchnetvlad \
#						--samples_per_class 2 --loss margin --batch_mining distance \
#						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
#						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
#						--append_pca_layer --num_pcs 128 --save_path Training_Results \
#						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
#						--evalevery 1 \
#						--resume_path /home/czhang/DIML/Training_Results/MSLS/Apr11_16-16-09_MSLS_CVT_finetune_triplet+KD10_1e-4/checkpoints/model_best.pth.tar \
#						--distill embed --kd_weight 10.0
						
			
# distill cvt from vlad (embed + triplet)			
CUDA_VISIBLE_DEVICES=1,2 python train_msls_KDembed_tri.py --dataset $dataset --kernels 8 --source_path $datapath --n_epochs 30 --group MSLS_CVT_triplet+KD10_1e-4 --loss_margin_beta 0.6 --seed 0 --bs 6 --pool patchnetvlad \
						--samples_per_class 2 --loss margin --batch_mining distance \
						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
						--append_pca_layer --num_pcs 128 --save_path Training_Results \
						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
						--evalevery 1 \
						--distill embed --kd_weight 10.0
