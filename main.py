from network import VDSR
from utils import *
import argparse

def check_phase(args):
    args.phase = args.phase.lower()
    assert args.phase == 'train' or args.phase == 'test', 'Choose TRAIN or TEST phase'

def check_channel(args):
    args.channel = args.channel.lower()
    assert args.channel == 'y' or args.channel == 'rgb' or args.channel == 'ycbcr', 'Color Channel Should be Y(in YCbCr) or RGB'
    if args.channel == 'ycbcr':
        args.num_channel = 3
    else:
        args.num_channel = len(args.channel)
    return args

def check_args(args):
    check_phase(args)
    args = check_channel(args)
    args.train_scale = str2int_list(args.train_scale)
    return args

def parse_args():
    desc = "Tensorflow implementation of VDSR"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test?')

    parser.add_argument('--channel', type=str, default='Y', help='RGB or Y or YCbCr')
    parser.add_argument('--depth', type=int, default=20, help='The number of nework depth')
    parser.add_argument('--train_scale', type=str, default='2,3,4', help='Training Super-Resolution Scale')
    parser.add_argument('--test_scale', type=int, default=2, help='Testing Super-Resolution Scale')

    parser.add_argument('--epoch', type=int, default=80, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch per gpu')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--grad_clip', type=float, default=0.001, help='the gradient clipping param')

    parser.add_argument('--val_interval', type=int, default=100, help='The validation interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    print(args)
    # quit()
    with tf.Session() as sess:
    
        cnn = VDSR(sess, args)
        cnn.build_model()
        show_all_variables()
    
        if(args.phase == 'train'):
            cnn.train()
        
        cnn.test()