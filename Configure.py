import argparse

def configure():
    parser = argparse.ArgumentParser()

  
    parser.add_argument("--mode", type=str, default='predict', help='train/test/predict modes')
    parser.add_argument("--batch_size", type=int, default=64, help='training batch size')
    parser.add_argument("--valid_list", type=list, default=[100,120,140,160,170,180,190,200], help='hyperparameter for epoch size')
    parser.add_argument("--max_epoch", type=int, default=200, help='max epoch size')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--learning_rate", type=float, default=0.01, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help='weight decay rate')
    parser.add_argument("--datadir", type=str, default='/content/drive/MyDrive/Colab/Ashutosh_Chauhan/data/', help='data directory')
    parser.add_argument("--privatedir", type=str, default='/content/drive/MyDrive/Colab/Ashutosh_Chauhan/Private/', help='private test directory')
    parser.add_argument("--modeldir", type=str, default='/content/drive/MyDrive/Colab/Ashutosh_Chauhan/model/parameters/', help='model directory')

    return parser.parse_args()

