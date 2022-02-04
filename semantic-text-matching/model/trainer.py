"""
Trainer class
"""
import time
import torch
from torch import nn
from sys import stdout
from os.path import join
from utils.logging import logger
from model.tester import Tester

default_save_path = "./models/"


class Trainer(object):
    def __init__(self, model=None, train_data=None, validation_set=None, optimizer=None, loss=None,
                 update_every=1, n_epochs=10, print_every=10, metrics=None, validate_every=3,
                 save_path=default_save_path, device=None):

        # checking whether the model type is correct
        if not isinstance(model, nn.Module):
            raise TypeError("Trainer requires model to be type nn.Module, but instead got {}!".format(type(model)))

        # checking metrics and validation data
        if (not metrics) and validation_set is not None:
            raise ValueError("No metric for the validation set!")
        if metrics and (validation_set is None):
            raise ValueError("No validation data for evaluations... Pass validation data, or set metrics to None!")

        # updating with gradient every ... steps
        assert update_every >= 1  # update every has to be at least 1
        self.update_every = int(update_every)

        self.model = model
        self.train_data = train_data
        self.validation_data = validation_set
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.metrics = metrics
        self.validate_every = validate_every
        self.device = device
        self.save_path = save_path
        self.tester = None
        self.best_validation_results = None

        if save_path is None:
            self.save_path = ""

        if validation_set is not None:
            self.tester = Tester(model=self.model, test_data=validation_set, metrics=self.metrics)

        if model:
            self.model.train()  # set model in training phase

        self.saved_model_name = "best_validation_model" + str(round(time.time())) + ".pth"  # creating a random name

    def train(self):
        logger.info("Starting training....")

        average_loss = 0.0  # average loss per every n batches
        for epoch in range(self.n_epochs):

            step = 0  # step counter
            epoch_loss = 0  # epoch loss
            for index, batch in enumerate(self.train_data):
                # moving tensors on the device
                text_id = batch["text_id"].to(self.device)
                candidate_id = batch["candidate_id"].to(self.device)
                summary_id = batch["summary_id"].to(self.device)

                # zero gradients
                self.optimizer.zero_grad()

                # calculate prediction
                pred = self.model(text_id, candidate_id, summary_id)

                # propagate loss backwards
                loss = self.loss(pred["candidate_scores"], pred["summary_score"])
                average_loss += loss.item()
                epoch_loss += loss.item()
                loss.backward()

                # update weights
                self.optimizer.step()

                del text_id, candidate_id, summary_id  # deleting used memory from the GPU

                if step % self.print_every == 0:
                    average_loss = float(average_loss) / self.print_every
                    stdout.write("\rEpoch: %d -- Step: %d -- Train loss: %f" % (epoch, step, average_loss))
                    average_loss = 0.0

                step += 1

            epoch_loss = float(epoch_loss) / step
            stdout.write("\r")  # to clear the output
            print("Epoch: " + str(epoch) + "  -- Loss: " + str(epoch_loss) + "            ")
            logger.info("Epoch %d -- Train loss: %f", epoch, epoch_loss)

            if epoch % self.validate_every == 0:
                if self.tester is not None:
                    eval_results = self.tester.test()  # validating the model

                    if self._check_results(eval_results, metric="ValidationMetric",
                                           field="ROUGE"):  # checking whether the results are currently the best
                        logger.info("Current best model is at epoch %d with train loss: %f", epoch, epoch_loss)
                        self._save_model()  # if we have the best results, we save the model!
                    print("Evaluation results at epoch " + str(epoch) + " :" + str(eval_results))
                    logger.info("Epoch %d -- Evaluation results: " + str(eval_results), epoch)

    def _check_results(self, evaluation_results, metric=None, field=None):

        if metric is None:
            raise ValueError("Metric is empty! There has to be a metric and field based on which you are "
                             "comparing models!")

        if self.best_validation_results is None:
            self.best_validation_results = evaluation_results
            return True

        if evaluation_results[metric][field] > self.best_validation_results[metric][field]:
            logger.info("Current BEST score for %s METRIC is %f!", field, evaluation_results[metric][field])
            self.best_validation_results = evaluation_results
            return True

        return False

    def _save_model(self):
        torch.save(self.model.state_dict(), join(self.save_path, self.saved_model_name))
