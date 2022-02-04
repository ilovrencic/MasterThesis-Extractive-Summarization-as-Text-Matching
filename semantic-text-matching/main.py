""""
#######################
      MAIN FILE
#######################


Example input:

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --mode=train --save_path=./bert_model --gpus=0,1 

"""
import argparse
from train import train_model
from test import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training or testing LoxSumm"
    )

    # REQUIRED FIELDS
    parser.add_argument("--mode", required=True, help="Choosing between training and testing mode", type=str)

    parser.add_argument("--save_path", required=True, help="Path where the model will be saved", type=str)

    parser.add_argument("--gpus", required=True, help="Available GPUs for training/testing",
                        type=str)  # example input => --gpus=0,1,2

    # OPTIONAL FIELDS
    parser.add_argument("--batch_size", default=2, help="Training batch size", type=int)

    parser.add_argument("--accum_count", default=2, help="Number of update steps before performing backward pass",
                        type=int)

    parser.add_argument("--candidate_num", default=20, help="Number of candidate summaries", type=int)

    parser.add_argument("--max_lr", default=0.001, help="Maximum learning rate", type=float)

    parser.add_argument("--margin", default=0.01, help="Parameter for MarginRankingLoss", type=float)

    parser.add_argument("--warmup_steps", default=10000, help="Warm up steps for training", type=int)

    parser.add_argument("--n_epochs", default=5, help="Total number of training epochs", type=int)

    parser.add_argument("--valid_every", default=2,
                        help="Number of update steps for validation and saving checkpoint", type=int)

    arguments = parser.parse_known_args()[0]

    if arguments.mode == "train":
        train_model(arguments)
    else:
        test_model(arguments)
