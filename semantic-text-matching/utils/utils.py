"""
#######################
	 UTILITIES
#######################
"""
import json
import os
from os.path import exists, join

"""
Method that returns data paths and checks whether they really exist
"""
def check_data_path(args):
    data_paths = {}

    if args.mode == "train":
        data_paths["train"] = "data/train_CNNDM_bert.jsonl"  # change this according to your own dataset
        data_paths["val"] = "data/val_CNNDM_bert.jsonl"  # change this according to your own dataset
    else:
        data_paths["test"] = "data/test_CNNDM_bert.jsonl"  # change this according to your own dataset

    for mode in data_paths:
        assert exists(data_paths[mode])

    return data_paths

"""
Method that paths for decoded and reference pairs

"""
def get_results_path(save_path, curr_model):
    result_path = join(save_path, "results")
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, curr_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, "dec")
    ref_path = join(model_path, "ref")
    os.makedirs(dec_path)
    os.makedirs(ref_path)
    return dec_path, ref_path

"""
Method that creates generator that reads sequential json datapoints from the file.

Arguments:
	- path = path to the json file
	- fields = fields in the file that are important
	- encoding = file encoding

"""
def read_json(path, fields=None, encoding='utf-8'):
    if fields is not None:
        fields = set(fields)

    with open(path, 'r', encoding=encoding) as f:
        for line_idx, line in enumerate(f):
            data = json.loads(line)

            if fields is None:
                yield line_idx, data, len(line)
                continue

            field_data = {}
            for key, value in data.items():
                if key in fields:
                    field_data[key] = value

            if len(field_data) < len(fields):
                raise ValueError("Invalid instance at line: {}".format(line_idx))

            yield line_idx, field_data, len(line)


"""
Method that prints current progress of certain task.
"""
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


"""
Method that saves last parameter configuration into a json file.
"""
def save_last_parameter_configuration(args):
    params = {'candidate_num': args.candidate_num, 'batch_size': args.batch_size, 'accum_count': args.accum_count,
              'max_lr': args.max_lr, 'margin': args.margin, 'warmup_steps': args.warmup_steps,
              'n_epochs': args.n_epochs, 'valid_steps': args.valid_every}

    with open('./parameters/params.json', 'w') as f:
        json.dump(params, f, indent=4)
