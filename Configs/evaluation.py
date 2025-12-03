'''
Define hyperparameters.
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters')

    # dataset
    parser.add_argument("--datadir", type=str, default="./Data", help="data directory")
    parser.add_argument("--dataset", type=str, default="Bearing", choices=["Bearing", "Gearbox", "Motor"])
    parser.add_argument("--load", type=int, default=0, help="working condition")
    parser.add_argument("--num_train", type=int, default=343, help="the number of samples per class")
    parser.add_argument("--num_validation", type=int, default=49, help="the number of validation samples per class")
    parser.add_argument("--num_test", type=int, default=98, help="the number of test samples per class")
    parser.add_argument("--labels", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12", help="training classes")
    parser.add_argument("--num_workers", type=int, default=2, help="the number of dataloader workers")
    parser.add_argument("--label_ratio", type=float, default=0.01, help="the ratio of labeled samples")

    # evaluation
    parser.add_argument("--mode", type=str, default="linear_protocol", choices=["linear_protocol", "semi_supervised"])
    parser.add_argument("--model", type=str, default="CoWS")
    parser.add_argument("--verbose", type=bool, default=True, help="whether print traing log")
    parser.add_argument("--max_epochs", type=int, default=40, help="the number of maximum of training epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--backbone_lr", type=float, default=1e-3)
    parser.add_argument("--classifier_lr", type=float, default=1e-1)

    args = parser.parse_args()

    return args