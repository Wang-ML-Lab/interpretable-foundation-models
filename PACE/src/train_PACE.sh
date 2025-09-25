CUDA_VISIBLE_DEVICES=0 python main.py --train  --task Color --name ViT-PACE --num_epochs 1 --pretrain_epoch 5 &
CUDA_VISIBLE_DEVICES=1 python main.py --train  --task flower102 --name ViT-PACE --num_epochs 1 --pretrain_epoch 10 &
CUDA_VISIBLE_DEVICES=0 python main.py --train  --task cub2011 --name ViT-PACE --num_epochs 1 --pretrain_epoch 10 &
CUDA_VISIBLE_DEVICES=1 python main.py --train  --task cars --name ViT-PACE --num_epochs 1 --pretrain_epoch 20 &