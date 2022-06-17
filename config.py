import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--cudaDevice', default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--featureExtract', action='store_true', help='train from Scratch')
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--lrX', type=float, default=0.1, help='learning rate for Subnets')
    parser.add_argument('--lrY', type=float, default=0.2, help='learning rate for Subnets')
    parser.add_argument('--lrZ', type=float, default=0.2, help='learning rate for Subnets')
    parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--num_workers', default=0)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--model_name', default='resnext50', help='resnet18, resnet50, resnext50, Attrresnext50')


    return parser
