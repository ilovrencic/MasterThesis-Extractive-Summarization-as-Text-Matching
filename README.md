# Master's Thesis: Deep Neural Models for Extractive Summarization

### Abstract

Large amounts of publicly available textual data made possible the rapid developments of neural natural language processing (NLP) models. One of the NLP tasks that particularly benefited from large amounts of text, but which at the same time holds promise for solving the problem of data overabundance, is automated text summarization. In particular, the goal of extractive text summarization is the production of a short but informationally rich and con- densed subset of the original text. The topic of this thesis is to research and (re)implement a new approach within the extractive summarization field, which focuses on framing extractive summarization as a semantic text-matching problem. As of now, most of the neural extractive summarization models follow the same paradigm: extract sentences, score them and pick the most salient ones. However, by choosing the most salient sentences, we are often left with a summary where most sentences are redundant. Zhong et al. (2020) proposed a novel paradigm where instead of choosing the salient sentences individually (sentence-level summarization), the focus is on simultaneously generating and picking the most salient summaries (summary-level summarization). The objective of the thesis is to reimplement this novel paradigm, research the flaws of previous models, and potentially improve the capabil- ities of this new summarization approach. All references must be cited, and all source code, documentation, executables, and datasets must be provided with the thesis.

### Model architecture

The model is split in two modules - **candidate generation module** and **semantic text matching module**. 

### How to recreate the experiments and how to use it? 

The details for model execution are detailed in the **main.py** file. 

## Authors

* **[Ivan Lovrenčić](https://github.com/ilovrencic)**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **[TakeLab](https://takelab.fer.hr/)**
* **[Tingyu Qu](https://www.kuleuven.be/wieiswie/en/person/00125529)**

