"""
######################################

	Testing file

#####################################
"""

import os
import torch
from os.path import join
from utils.utils import check_data_path, save_last_parameter_configuration
from dataloader import SummaryDataIterator
from model.metrics import MatchRougeMetric
from model.tester import Tester
from model.model import SummModel
from utils.logging import logger, init_logger
from utils.utils import get_results_path

logging_path = "./logs/summary.log"  # path to log files


def test_model(args):
    # check data paths that contain the datasets
    data_paths = check_data_path(args)

    # initialize logger
    init_logger(log_file=logging_path)

    # load summarization datasets
    datasets = SummaryDataIterator(paths=data_paths, fields=["text_id", "summary_id", "candidate_id"],
                                   batch_size=1).get_iterators()  # because of the metric, batch size has to be one!
    test_set = datasets["test"]

    # configure training
    devices = [int(gpu) for gpu in args.gpus.split(',')]
    save_last_parameter_configuration(args)
    logger.info("Number of available devices for training {}!".format(len(devices)))

    models = os.listdir(args.save_path)  # list of all models we have saved

    if not torch.cuda.is_available():
        raise Exception("CUDA is necessary for Model testing!")

    device = torch.device("cuda")  # GPU is available!
    test_set_local = test_set.get_dataset_locally()

    for curr_model in models:
        if not curr_model.endswith('.pth'):
            continue

        print("Current model: " + str(curr_model))
        logger.info("Current model for testing is: %s!", curr_model)

        model = torch.nn.DataParallel(SummModel())
        model.to(device)

        model.load_state_dict(torch.load(join(args.save_path, curr_model)))
        model.eval()

        dec_path, ref_path = get_results_path(args.save_path, curr_model)
        test_metric = [
            MatchRougeMetric(data=test_set_local, dec_path=dec_path, ref_path=ref_path, n_total=len(test_set_local))]
        tester = Tester(model=model, test_data=test_set, metrics=test_metric)
        results = tester.test()

        logger.info("Test results: " + str(results))
        print("Results: " + str(results))
