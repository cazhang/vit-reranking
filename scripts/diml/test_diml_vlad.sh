dataset=MSLS
arch=netvlad_pca128
datapath=/home/czhang/Drive2/datasets/revisit_dml

CUDA_VISIBLE_DEVICES=0 python test_msls_vlad_diml.py --dataset $dataset \
              --source_path $datapath \
              --seed 0 --bs 8 --data_sampler class_random --samples_per_class 2 \
              --arch $arch --group diml_test_res50 \
              --embed_dim 128 --evaluate_on_gpu \
              --pooling netvlad \
              --num_clusters 16 \
						  --vlad_ckpt /home/czhang/Pretrained_models/Netvlad/mapillary_WPCA128.pth.tar \
						  --append_pca_layer --num_pcs 128