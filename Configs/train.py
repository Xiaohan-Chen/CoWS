'''
Define hyperparameters.
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters')

    # dataset
    parser.add_argument("--datadir", type=str, default="./Data", help="data directory")
    parser.add_argument("--dataset", type=str, default="Bearing", choices=["Bearing"])
    parser.add_argument("--load", type=int, default=0, help="working condition")
    parser.add_argument("--num_train", type=int, default=343, help="the number of samples per class")
    parser.add_argument("--num_validation", type=int, default=49, help="the number of validation samples per class")
    parser.add_argument("--num_test", type=int, default=98, help="the number of test samples per class")
    parser.add_argument("--labels", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12", help="training classes")
    parser.add_argument("--num_workers", type=int, default=2, help="the number of dataloader workers")

    # training
    parser.add_argument("--max_epochs", type=int, default=400, help="the number of maximum of training epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--print_step", type=int, default=10, help="print training log every x epochs")
    parser.add_argument("--dim", type=int, default=256, help="hidden dimension")

    args = parser.parse_args()

    return args