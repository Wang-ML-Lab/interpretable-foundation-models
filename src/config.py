import argparse
parser = argparse.ArgumentParser(description='PACE')

# data args
parser.add_argument('--data_path', type=str, help='path of dataset', 
                    default='../dataset')
parser.add_argument('--task',type=str,help='task name of dataset',default='toy') 
parser.add_argument('--save_path', type=str, help='path to save', 
                    default='../ckpt')
parser.add_argument('--load_path', type=str, help='path to load', 
                    default='../ckpt')

# model args

## for PACE
parser.add_argument('--c_dim', type=int, help='dimension of PACE',default=25)
parser.add_argument('--K', type=int, help='number of centers of PACE',default=100) 
parser.add_argument('--D', type=int, help='number of images',default=10000)
parser.add_argument('--N', type=int, help='max length of images',default=197)
parser.add_argument('--alpha', type=float, help='alpha prior of PACE',default=2) 
parser.add_argument('--eta', type=float, help='weight of PACE',default=1)
parser.add_argument('--frac', type=str,help='type of fractional model', default='fix') 
parser.add_argument('--layer', type=int, help='layer of PACE',default=-2) 
parser.add_argument('--version', type=str, help='running version',default='v0')

## for ViT
parser.add_argument('--lm', type=str, help='which language model', default='ViT') 
parser.add_argument('--b_dim', type=int, help='dimension of ViT',default=768)      
parser.add_argument('--out_dim', type=int, help='dimension of output',default=5)                    
parser.add_argument('--name', type=str, help='model name',  default='ViT-PACE')              
parser.add_argument('--seed',type=int, default=2021)
parser.add_argument('--lr', type=float, help='learning rate', default=3e-5)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.05)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--require_grad', action='store_true', default=False)

# optimization args
parser.add_argument('--num_epochs', type=int, help='number of epoches',default=10)
parser.add_argument('--train_batch_size', type=int, help='training sz',default=16)
parser.add_argument('--eval_batch_size', type=int, help='eval sz',default=64)                    
parser.add_argument('--metric', type=str, help='eval metric',default='eval_accuracy')           
              




    
