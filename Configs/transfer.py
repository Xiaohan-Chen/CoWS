'''
Define hyperparameters.
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters')

    # dataset
    parser.add_argument("--datadir", type=str, default="./Data", help="data directory")
    parser.add_argument("--src_data", type=str, default="Bearing", choices=["Bearing"])
    parser.add_argument("--src_load", type=int, default=0, help="working condition")
    # target domain data:
    parser.add_argument("--tgt_data", type=str, default="HUSTBearing", choices=["HUSTBearing", "HITBearing", "XJTUBearing"])
    parser.add_argument("--tgt_load", type=int, default=3, help="working condition")
    parser.add_argument("--num_train", type=int, default=10, help="the number of samples per class in target domain")
    parser.add_argument("--num_validation", type=int, default=100, help="the number of validation samples per class in target domain")
    parser.add_argument("--num_test", type=int, default=100, help="the number of test samples per class in target domain")
    parser.add_argument("--labels", type=str, default="0,1,2,3,4", help="training classes")
    parser.add_argument("--num_workers", type=int, default=2, help="the number of dataloader workers")

    # evaluation
    parser.add_argument("--model", type=str, default="CoWS", help="model name")
    parser.add_argument("--verbose", type=bool, default=False, help="whether print traing log")
    parser.add_argument("--max_epochs", type=int, default=40, help="the number of maximum of training epoch")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--backbone_lr", type=float, default=1e-3)
    parser.add_argument("--classifier_lr", type=float, default=1e-2)

    args = parser.parse_args()

    return args