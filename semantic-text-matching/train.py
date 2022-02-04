"""
######################################

	Training and validation file

#####################################
"""
from utils.utils import check_data_path, save_last_parameter_configuration
from utils.logging import init_logger, logger, reset_log
from dataloader import SummaryDataIterator
from model.model import SummModel
from model.trainer import Trainer
from model.trainer import Tester
from torch.optim import Adam, SGD
from model.metrics import MarginRankingLoss, ValidationMetric
import torch

logging_path = "./logs/summary.log"  # path to log files


def train_model(args):
    # check data paths that contain the datasets
    data_paths = check_data_path(args)

    # initialize logger
    init_logger(log_file=logging_path)

    # load summarization datasets
    datasets = SummaryDataIterator(paths=data_paths, fields=["text_id", "summary_id", "candidate_id"],
                                   batch_size=args.batch_size, candidate_num=args.candidate_num).get_iterators()
    train_set = datasets["train"]
    val_set = datasets["val"]

    # configure training
    devices = [int(gpu) for gpu in args.gpus.split(',')]
    save_last_parameter_configuration(args)
    logger.info("Number of available devices for training {}!".format(len(devices)))

    # model configuration
    if len(devices) == 1 and torch.cuda.device_count() == 1:  # if there only one available gpu
        model = SummModel()
        device = torch.device("cuda")
        model.to(device)
        logger.info("Using only one GPU!")
    elif len(devices) > 1 and torch.cuda.device_count() > 1:  # there is more gpus available
        model = SummModel()
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model, devices)
        model.to(device)
        logger.info("Using two GPUs!")
    else:  # there is no gpu available, so we use cpu
        model = SummModel()
        device = "cpu"
        model.to(device)
        logger.info("Using CPU!")

    # training configuration
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr)
    loss = MarginRankingLoss(margin=args.margin)
    metric = [ValidationMetric(data=val_set.get_dataset_locally())]

    # initializing trainer and starting training
    trainer = Trainer(model=model, train_data=train_set, validation_set=val_set, loss=loss, optimizer=optimizer,
                      device=device, metrics=metric, save_path=args.save_path, n_epochs=args.n_epochs,
                      validate_every=args.valid_every)
    trainer.train()
