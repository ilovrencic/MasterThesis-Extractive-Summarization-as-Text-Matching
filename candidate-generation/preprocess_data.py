import os
import json
import argparse
import progressbar
import functools
import multiprocessing as mp
from rouge import Rouge
from itertools import combinations
from transformers import BertTokenizer
from os.path import exists

BERT_MAX_LEN = 512

"""
CANDIDATE GENERATION MODULE
"""


"""
Method that loads the data from the
path that was passed. The data is loaded into
a list.

Arguments:
	- path = path to the JSON file that contains some data
"""
def load_data(path):
	data = []
	with open(path) as file:
		for line in file:
			data.append(json.loads(line))
	return data


"""
Method that calculates the average rouge score between the candidate and the reference (gold summary).
The average is taken among ROUGE-1, ROUGE-2 and ROUGE-L measure.
"""
def calc_rouge_score(candidate, summary):
	rouge = Rouge()

	dec = "".join(sent for sent in candidate)
	ref = "".join(sent for sent in summary)

	scores = rouge.get_scores(dec,ref)
	return (scores[0]["rouge-1"]["f"]+scores[0]["rouge-2"]["f"]+scores[0]["rouge-l"]["f"])/3


"""
Method that preprocess a single datapoint. Each datapoint is preprocessed and the stored in 
a temporary file that's later going to be read and then written in a single file.

Arugments:
	- current_datapoint - dictonary with original text and gold summary
	- sent_id - list of indexes with positions of top N salient sentances
	- tokenzier - BERT tokenizer
	- candidate_sentances - number of candidate sentances that are taken into consideration
"""
def preprocess_single(original_data,index_data,tokenizer,candidate_sentances,temp_path, index):

	current_datapoint = {}
	current_datapoint["text"] = original_data[index]["text"]
	current_datapoint["summary"] = original_data[index]["summary"]

	id = index

	sent_id = index_data[index]["sent_id"]

	cls, sep = '[CLS]','[SEP]'
	sep_id = tokenizer.encode(sep, add_special_tokens = False)

	preprocessed_data = {}
	preprocessed_data["text"] = current_datapoint["text"]
	preprocessed_data["summary"] = current_datapoint["summary"]

	assert len(sent_id) >= candidate_sentances
	sent_id = sent_id[:candidate_sentances]

	preprocessed_data["ext_idx"] = sent_id

	# How to create candidate summaries? 
	# 1. candidate_sentances represenets a parameter that takes first N of the indexes of top salient sentances. 
	# E.g. if sent_id = [1, 2, 5, 0, 3] and candidate sentances = 3, then the results is [1, 2, 5]
	# 2. Create different combinations of summaries with the candidate sentances
	# E.g. if candidate sentaces = 5, and sent_id = [1, 2, 5, 4, 0], we combinations of (5 2) and (5 3). We 
	# choose 2 and 3, because summaries are usually around 2 to 3 sentances long, but this can be changed! 
	# 3. Create candidates and score them
	index_combinations = list(combinations(sent_id,2))
	index_combinations += list(combinations(sent_id,3))

	if len(sent_id) < 2:
		index_combinations = sent_id

	#creating and scoring candidate summaries
	scores = []
	for indexes in index_combinations:
		indexes = list(indexes)
		indexes.sort()

		candidate = []
		for index in indexes:
			sent = current_datapoint["text"][index]
			candidate.append(sent)
		scores.append((indexes,calc_rouge_score(candidate, current_datapoint["summary"])))

	scores.sort(key=lambda x: x[1], reverse=True)

	preprocessed_data["indices"] = []
	preprocessed_data["score"] = []

	cand_ids = []
	cand_scores = []
	for indexes, score in scores:
		preprocessed_data["indices"].append(list(indexes))
		preprocessed_data["score"].append(score)
	
	#Creating the candidates
	candidate_summaries = []
	for indexes in preprocessed_data["indices"]:
		current_candidate = [cls]
		
		for index in indexes:
			current_candidate += current_datapoint["text"][index].split()

		#limiting the size of candidate to the size of input in BERT
		current_candidate = current_candidate[:BERT_MAX_LEN]
		current_candidate = ' '.join(current_candidate)
		candidate_summaries.append(current_candidate)

	preprocessed_data["candidate_id"] = []

	#Tokenizing and encoding the candidate summaries
	for candidate in candidate_summaries:
		tokens = tokenizer.encode(candidate, add_special_tokens = False)[:(BERT_MAX_LEN-1)]
		tokens += sep_id
		preprocessed_data["candidate_id"].append(tokens)


	preprocessed_data["text_id"] = []

	#Tokenizing and encoding the original text
	original_text = [cls]
	for sentance in current_datapoint["text"]:
		original_text += sentance.split()

	original_text = original_text[:BERT_MAX_LEN]
	original_text = ' '.join(original_text)
	tokens = tokenizer.encode(original_text[:BERT_MAX_LEN], add_special_tokens = False)[:(BERT_MAX_LEN-1)]
	tokens += sep_id
	preprocessed_data["text_id"] = tokens


	preprocessed_data["summary_id"] = []

	#Tokenizing and encoding the summary
	summary = [cls]
	for sentance in current_datapoint["summary"]:
		summary += sentance.split()
	summary = summary[:BERT_MAX_LEN]
	summary = ' '.join(summary)
	tokens = tokenizer.encode(summary, add_special_tokens = False)[:(BERT_MAX_LEN-1)]
	tokens += sep_id
	preprocessed_data["summary_id"] = tokens

	#writing the preprocessed datapoint into temporary file
	with open(temp_path+"/{}.jsonl".format(id),"a") as f:
		f.write(json.dumps(preprocessed_data))

"""
Main function of this script

The script does the following:
1.) Loads original data (documents and correspoding gold summaries) 
and indices (list of indexes that corresponds to top n most salient senteces)

2.) Concurrently preprocess each document 

	2.a) Generate candidates 

	2.b) Tokenize document, gold summary and candidates 

	2.c) Score them based on average ROUGE score and sort descendingly

	2.d) Write the result into a temporary file

3.) After we preprocessed all documents, go through 
all temporary files and merge them into one file
"""
def preprocess(arguments):

	#initialize the BERT tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	
	#loading the data from the paths
	original_data = load_data(arguments.original_path)
	index_data = load_data(arguments.index_path)
	data_size = len(original_data)

	#checking whether there is same amount of data
	assert len(original_data) == len(index_data)
	print("Ready to preprocess {} documents!".format(len(original_data)))

	#Creating a temporary directory for preprocessed files
	temp_path = os.getcwd()+"/tmp"
	os.mkdir(temp_path)

	#Creating a bar object that's going to measure the progress
	bar = progressbar.ProgressBar(maxval=len(original_data*2), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) # <-- remove object Bar if you want to remove progressbar dependency from the file! 
	bar.start()

	counter = 0

	#Multiprocess the work of preprocessing the single datapoint in the dataset
	with mp.Pool() as pool:
		for i, _ in enumerate(pool.imap_unordered(functools.partial(preprocess_single, original_data, index_data, tokenizer, arguments.cand_comb,temp_path),range(data_size)),1):
			bar.update(counter)
			counter += 1

	#going through temporary files and writing preprocessed datapoints into a single file
	for index in range(len(original_data)):
		with open(temp_path+"/{}.jsonl".format(index)) as f:
			data = json.loads(f.read())
		with open(arguments.preprocessed_path, 'a') as f:
			print(json.dumps(data), file = f)

		bar.update(counter)
		counter += 1

	bar.finish()
	print("Finished preprocessing the dataset. The preprocessed data is in: {}".format(arguments.preprocessed_path))
	os.system('rm -r {}'.format(temp_path))



"""
----------- Preprocessing script -------------
This file serves as a script that preprocesses
the original dataset into the more manageable 
preprocessed dataset. To utilize this script, 
the following is required - the path to the 
original dataset, the path to the file with 
indexes of top N sentences, and the path where 
the preprocessed dataset will be saved. 
----------------------------------------------

Arguments: 
	- original dataset path 
	- file with indexes of top N sentances per article path
	- preprocessed dataset path

Original dataset format: 
	- JSON
	- "text" and "summary" fields

Preprocessed dataset format:
	- JSON
	- "text","summary", "indexes", "cand_comb","cand_scores","candidate_ids","text_ids" and "summary_ids" fields

Index file format:
	- JSON
	- "sentance_ids" field

"""
if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description = "Preprocess the original dataset and create candidate summaries"
	)


	#Original dataset path 
	parser.add_argument("--original_path", type = str, required = True)

	#Path with file that contains indexes of top N sentances for each datapoint	
	parser.add_argument("--index_path", type = str, required = True)

	#Preprocessed dataset path
	parser.add_argument("--preprocessed_path", type = str, required = True)

	#Number of candidate sentances
	parser.add_argument("--cand_comb", type = int, default = 5)

	arguments = parser.parse_args()
	assert exists(arguments.original_path)
	assert exists(arguments.index_path)
	
	print("Preprocessing has started!")
	preprocess(arguments)	