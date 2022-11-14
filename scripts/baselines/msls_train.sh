# """============= Baseline Runs --- MSLS ===================="""
main=train_msls_baseline
dataset=MSLS
datapath=/home/czhang/Drive2/datasets/revisit_dml


## cvt baseline
#CUDA_VISIBLE_DEVICES=1,2 python $main.py --dataset $dataset --kernels 8 --source_path $datapath --bs 4 #--cachebatchsize 20 --optim Adam --lr 0.0001 --lrstep 5 --lrgamma 0.5 --n_epochs 30 \
#            --weightdecay 0.001 --momentum 0.9 --patience 5 --evalevery 1 --margin 0.1 \
#            --nNeg 5 --imageresizew 640 --imageresizeh 480 --grid_size 7 \
#						--group MSLS_CVT_adam --loss_margin_beta 0.6 --seed 0 \
#						--samples_per_class 2 --loss margin --batch_mining distance \
#						--arch cvt_13_normalize --embed_dim 128


# ir resnet 50 baseline
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 8 --source_path $datapath --bs 2 #--cachebatchsize 20 --optim adam --lr 0.0001 --lrstep 5 --lrgamma 0.5 --n_epochs 30 \
#            --weightdecay 0.001 --momentum 0.9 --patience 5 --evalevery 1 --margin 0.1 \
#            --nNeg 5 --imageresizew 640 --imageresizeh 480 --grid_size 7 \
	#					--group MSLS_Res50_d512 --loss_margin_beta 0.6 --seed 0 \
	#					--samples_per_class 2 --loss margin --batch_mining distance \
	#					--arch irresnet50_gem --embed_dim 512
						
# distill cvt from vlad (triplet info)
#CUDA_VISIBLE_DEVICES=1,2 python train_msls_distill.py --dataset $dataset --kernels 8 --source_path $datapath #--n_epochs 30 --group MSLS_CVT_distill_rand_1e-5 --loss_margin_beta 0.6 --seed 0 --bs 8 --pool patchnetvlad \
#						--samples_per_class 2 --loss margin --batch_mining distance \
#						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
#						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
#						--append_pca_layer --num_pcs 128 --save_path Training_Results \
#						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.00001 --cachebatchsize 20 \
#						--evalevery 1 --rand_distill --mini_data
						
# distill cvt from vlad (embeding )
CUDA_VISIBLE_DEVICES=1 python train_msls_KDembed.py --dataset $dataset --kernels 8 --source_path $datapath --n_epochs 30 --group MSLS_CVT_distill_embed_1e-4 --loss_margin_beta 0.6 --seed 0 --bs 16 --pool patchnetvlad \
						--samples_per_class 2 --loss margin --batch_mining distance \
						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
						--append_pca_layer --num_pcs 128 --save_path Training_Results \
						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
						--evalevery 1 --distill embed
						
# distill cvt from vlad (embeding + triplet )
CUDA_VISIBLE_DEVICES=2 python train_msls_baseline.py --dataset $dataset --kernels 8 --source_path $datapath --n_epochs 30 --group MSLS_CVT_finetune_triplet_1e-4 --loss_margin_beta 0.6 --seed 0 --bs 16 --pool patchnetvlad \
						--samples_per_class 2 --loss margin --batch_mining distance \
						--arch cvt_13_normalize --embed_dim 128 --num_clusters 16 \
						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
						--append_pca_layer --num_pcs 128 --save_path Training_Results \
						--imageresizew 224 --imageresizeh 224 --optim adam --lr 0.0001 --cachebatchsize 20 \
						--evalevery 1 --resume_path /home/czhang/DIML/Training_Results/MSLS/Apr08_15-57-57_MSLS_CVT_distill_embed_1e-4/checkpoints/model_best.pth.tar
						
					