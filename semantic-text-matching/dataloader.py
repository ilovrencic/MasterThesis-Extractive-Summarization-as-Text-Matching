"""
####################
    DataLoader
####################
"""
from utils.logging import logger
import pandas as pd
import torch

pd.options.mode.chained_assignment = None

"""
Dataloader class is used for loading the summary datasets from JSON files. 

- Suggested use: For smaller dataset ( < 2GB ), since the whole dataset will be loaded into memory.

Parameters:
    - fields_list = represents a list of fields that are important in JSON datapoints
    - max_len = max length of the candidate summary
"""
class SummaryDataLoader:
    def __init__(self, fields_list, max_len=180):
        self.fields_list = fields_list
        self.max_len = max_len
        self.sep_id = 102  # BERT '[SEP]'
        self.pad_id = 0  # BERT pad value

    """
    Method that loads the dataset and truncates and pads all the fields.
    
    """

    def _load(self, path):
        dataset = pd.read_json(path_or_buf=path, lines=True)

        def truncate_and_pad_summaries(summary):
            trunc_summary = summary
            if len(summary) > self.max_len:
                trunc_summary = summary[:(self.max_len - 1)]
                trunc_summary.append(self.sep_id)
            else:
                while len(trunc_summary) < self.max_len - 1:
                    trunc_summary.append(self.pad_id)
            return trunc_summary

        def truncate_and_pad_candidates(candidates):
            trunc_candidates = []
            for candidate in candidates:
                trunc_candidate = candidate
                if self.max_len < len(candidate):
                    trunc_candidate = candidate[:(self.max_len - 1)]
                    trunc_candidate.append(self.pad_id)  # adding separator id
                else:
                    while len(trunc_candidate) < self.max_len - 1:
                        trunc_candidate.append(self.sep_id)
                trunc_candidates.append(trunc_candidate)
            return trunc_candidates

        def pad_text_id(text_id):
            trunc_text = text_id
            bert_max_len = 512
            if len(text_id) >= bert_max_len:
                trunc_text = text_id[:(bert_max_len - 1)]
                trunc_text.append(self.sep_id)
            else:
                while len(trunc_text) < bert_max_len - 1:
                    trunc_text.append(self.pad_id)
            return trunc_text

        for field in self.fields_list:
            if field == "text_id":
                dataset[field] = dataset[field].apply(pad_text_id)
            elif field == "summary_id":
                dataset[field] = dataset[field].apply(truncate_and_pad_summaries)
            else:
                dataset[field] = dataset[field].apply(truncate_and_pad_candidates)

        return dataset[self.fields_list]

    def load(self, paths):
        print("Starting loading the dataset...")
        datasets = {}

        for mode in paths:
            print("Loading {} dataset...".format(mode))
            datasets[mode] = self._load(paths[mode])

        return datasets

    def load_from_path(self, paths):
        return self.load(paths)


class SummaryDataset(object):
    def __init__(self, path, fields, batch_size, max_length, candidate_num):
        self.path = path
        self.fields = fields
        self.max_len = max_length
        self.batch_size = batch_size
        self.candidate_num = candidate_num
        self.sep_id = 102
        self.pad_id = 0

        self.iterator = self._create_iterator()

    # using this will go through whole dataset and store it on the RAM
    def get_dataset_locally(self):
        dataset = {}
        self.iterator = self._create_iterator(chunksize=1)  # in order to get the full dataset, but not in chunks

        for index, batch in enumerate(self.iterator):
            dataset[index] = batch

        self.iterator = self._create_iterator()
        return dataset

    def _create_iterator(self, chunksize=None):
        if chunksize:
            return pd.read_json(path_or_buf=self.path, lines=True, chunksize=chunksize)
        else:
            return pd.read_json(path_or_buf=self.path, lines=True, chunksize=self.batch_size)

    def _transform_batch(self, batch):
        trans_batch = {}
        for i in range(len(batch)):
            for field in self.fields:
                if field == "text_id":
                    curr_value = batch.iloc[i].text_id
                elif field == "summary_id":
                    curr_value = batch.iloc[i].summary_id
                else:
                    curr_value = batch.iloc[i].candidate_id

                if field not in trans_batch:
                    trans_batch[field] = []

                trans_batch[field].append(curr_value)

        for key in trans_batch:
            trans_batch[key] = torch.tensor(trans_batch[key])

        return trans_batch

    def _error_wrapper(self, iterator):
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break
            except Exception as e:
                logger.info("Exception has occured during parsing the line! Exception: " + str(e))

    def __iter__(self):

        def truncate_and_pad_summaries(summary):
            trunc_summary = summary
            if len(summary) > self.max_len:
                trunc_summary = summary[:(self.max_len - 1)]
                trunc_summary.append(self.sep_id)
            else:
                while len(trunc_summary) < self.max_len:
                    trunc_summary.append(self.pad_id)
            return trunc_summary

        def truncate_and_pad_candidates(candidates):
            trunc_candidates = []
            for candidate in candidates:
                trunc_candidate = candidate
                if self.max_len < len(candidate):
                    trunc_candidate = candidate[:(self.max_len - 1)]
                    trunc_candidate.append(self.sep_id)  # adding separator id
                else:
                    while len(trunc_candidate) < self.max_len:
                        trunc_candidate.append(self.pad_id)
                trunc_candidates.append(trunc_candidate)
            return trunc_candidates

        def pad_text_id(text_id):
            trunc_text = text_id
            bert_max_len = 512
            if len(text_id) >= bert_max_len:
                trunc_text = text_id[:(bert_max_len - 1)]
                trunc_text.append(self.sep_id)
            else:
                while len(trunc_text) < bert_max_len:
                    trunc_text.append(self.pad_id)
            return trunc_text

        while True:
            for index, batch in enumerate(self._error_wrapper(self.iterator)):
                batch = batch[self.fields]

                for field in self.fields:
                    if field == "text_id":
                        batch[field] = batch[field].apply(pad_text_id)
                    elif field == "summary_id":
                        batch[field] = batch[field].apply(truncate_and_pad_summaries)
                    else:
                        batch[field] = batch[field].apply(truncate_and_pad_candidates)

                try:
                    transformed_batch = self._transform_batch(batch)
                    yield transformed_batch
                except Exception as e:
                    logger.error("Message occured at batch " + str(index) + "! Error: " + str(e))
            break
        self.iterator = self._create_iterator()  # create iterator again, so that we can do multiple epochs over dataset


"""
Data iterator that encapsulates Pandas dataframe.

- Suitable for larger datasets ( >= 2GB ) as data is loaded in chunks/batches.
"""
class SummaryDataIterator(object):
    def __init__(self, paths, fields, batch_size=15, candidate_length=180, candidate_num=20):
        self.paths = paths
        self.fields = fields
        self.candidate_length = candidate_length
        self.batch_size = batch_size
        self.candidate_num = candidate_num
        self.iterators = {}

        self._create_iterators()

    def _create_iterators(self):
        for mode in self.paths:
            self.iterators[mode] = SummaryDataset(path=self.paths[mode], fields=self.fields, batch_size=self.batch_size,
                                                  max_length=self.candidate_length, candidate_num=self.candidate_num)

    def get_iterators(self):
        return self.iterators
