"""""
Tester class 

Encapsulates the whole test process for MatchSum model
"""""
from torch import nn
from utils.logging import logger


class Tester(object):
    def __init__(self, model=None, test_data=None, metrics=None, batch_size=16, device=None):

        # checking whether the model type is correct
        if not isinstance(model, nn.Module):
            raise TypeError("Trainer requires model to be type nn.Module, but instead got {}!".format(type(model)))

        self.model = model
        self.test_data = test_data
        self.batch_size = batch_size
        self.metrics = metrics
        self.device = device

        if model:
            model.eval()  # set model into evaluation mode

    def test(self):
        logger.info("Starting testing....")
        eval_results = {}

        for index, batch in enumerate(self.test_data):
            # moving tensors on the device
            text_id = batch["text_id"].to(self.device)
            candidate_id = batch["candidate_id"].to(self.device)
            summary_id = batch["summary_id"].to(self.device)

            # getting prediction from model
            prediction = self.model(text_id, candidate_id, summary_id)

            del text_id, candidate_id, summary_id

            for metric in self.metrics:
                metric(prediction["candidate_scores"])

            del prediction # to save memory

        for metric in self.metrics:
            eval_result = metric.get_results()
            eval_results[metric.metrics_name()] = eval_result

        return eval_results
