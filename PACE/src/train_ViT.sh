CUDA_VISIBLE_DEVICES=1 python main.py --train  --task Color --name ViT-base --num_epochs 5 --lr 1e-3 --require_grad
CUDA_VISIBLE_DEVICES=1 python main.py --train  --task flower102 --name ViT-base --num_epochs 10 --lr 1e-3 --require_grad
CUDA_VISIBLE_DEVICES=1 python main.py --train  --task cub2011 --name ViT-base --num_epochs 10 --lr 1e-3 --require_grad
CUDA_VISIBLE_DEVICES=1 python main.py --train  --task cars --name ViT-base --num_epochs 20 --weight_decay 0.01 --lr 5e-5 --require_grad --seed 2025 --train_batch_size 32 --eval_batch_size 64

