from models import *
import argparse
from thop import profile
import torch


def model_params_flops(arch):
    print('==========================================================================')
    model = globals()[arch](num_classes=200)
    input = torch.randn(1, 3, 128, 128)
    macs, params = profile(model, inputs=(input,))
    print('==========================================================================')
    print('Total params:: {:.3f} M\n'
          'Total FLOPs: {:.3f}MFLOPs'.format(params/10**6, macs/10**6))
    print('==========================================================================')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the parameters and FLOPs of model')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet35',
    #                     help='model architecture: resnet35')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet44',
                        help='model architecture: resnet44')
    global args
    args = parser.parse_args()
    model_params_flops(args.arch)

