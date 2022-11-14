# """============= Baseline Runs --- MSLS ===================="""
main=test_msls_baseline
dataset=MSLS
datapath=/home/czhang/Drive2/datasets/revisit_dml

## vlad baseline
CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 8 --source_path $datapath --n_epochs 150 --group MSLS_VLAD_4096 --loss_margin_beta 0.6 --seed 0 --bs 8 --samples_per_class 2 --loss margin --batch_mining distance --arch netvlad_pca512 --embed_dim 4096 --num_clusters 16 --pooling patchnetvlad --vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA4096.pth.tar --append_pca_layer --num_pcs 4096

## vlad+pca baseline
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 8 --source_path $datapath --n_epochs 150 --group MSLS_VLAD --loss_margin_beta 0.6 --seed 0 --bs 24 \
#						--samples_per_class 2 --loss margin --batch_mining distance \
#						--arch netvlad_pca128 --embed_dim 128 --num_clusters 16 --pooling patchnetvlad \
#						--vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
#						--append_pca_layer --num_pcs 128 --save_path Test_Results \
#						--imageresizew 640 --imageresizeh 480

## cvt imgnet baseline
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 8 --source_path $datapath --n_epochs 150 \
#						--group MSLS_CVT_Imagenet --loss_margin_beta 0.6 --seed 0 --bs 24 \
#						--samples_per_class 2 --loss margin --batch_mining distance \
#						--arch cvt_inet_normalize_128 --embed_dim 384 --not_pretrained --num_classes 0 \
#						--imageresizew 224 --imageresizeh 224

## cvt baseline + diml 
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 8 --source_path $datapath --bs 16 #--cachebatchsize 20 --optim Adam --lr 0.00001 --lrstep 5 --lrgamma 0.5 --n_epochs 30 \
#            --weightdecay 0.001 --momentum 0.9 --patience 5 --evalevery 1 --margin 0.1 \
 #           --nNeg 5 --imageresizew 224 --imageresizeh 224 --grid_size 7 \
	#					--group MSLS_CVT_224_d128 --loss_margin_beta 0.6 --seed 0 \
	#					--samples_per_class 2 --loss margin --batch_mining distance \
	#					--arch cvt_13_normalize --embed_dim 128 --resume_path /home/czhang/DIML/Training_Results/MSLS/Mar18_23-30-54_MSLS_CVT_224_d128/checkpoints/model_best.pth.tar \
	#					--save_path Test_Results
	
	
## cvt distill + diml 
#CUDA_VISIBLE_DEVICES=0 python $main.py --dataset $dataset --kernels 8 --source_path $datapath --bs 16 #--cachebatchsize 20 --optim Adam --lr 0.00001 --lrstep 5 --lrgamma 0.5 --n_epochs 30 \
#            --weightdecay 0.001 --momentum 0.9 --patience 5 --evalevery 1 --margin 0.1 \
 #           --nNeg 5 --imageresizew 224 --imageresizeh 224 --grid_size 7 \
	#					--group MSLS_CVT_distill_224_d128 --loss_margin_beta 0.6 --seed 0 \
	#					--samples_per_class 2 --loss margin --batch_mining distance \
	#					--arch cvt_13_normalize --embed_dim 128 --resume_path /home/czhang/DIML/Training_Results/MSLS/Apr11_16-16-09_MSLS_CVT_finetune_triplet+KD10_1e-4/checkpoints/model_best.pth.tar \
	#					--save_path Test_Results --save_freq 10 --vis_freq 10
	
	
	


