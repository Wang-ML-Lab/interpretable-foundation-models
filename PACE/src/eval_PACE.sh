CUDA_VISIBLE_DEVICES=0 python run_quantitative_theta.py --task Color --name ViT-PACE --num_epochs 1 &
CUDA_VISIBLE_DEVICES=1 python run_quantitative_theta.py --task flower102 --name ViT-PACE --num_epochs 1 &
CUDA_VISIBLE_DEVICES=2 python run_quantitative_theta.py --task cub2011 --name ViT-PACE --num_epochs 1 &
CUDA_VISIBLE_DEVICES=3 python run_quantitative_theta.py --task cars --name ViT-PACE --num_epochs 1