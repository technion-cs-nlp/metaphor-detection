# Metaphor-Detection-Piyyut
A Dataset for Metaphor Detection in Early Medieval Hebrew Poetry

This repository contains the dataset for the paper "Metaphor Detection in Early Medieval Hebrew Poetry" by 
Michael Toker, Oren Mishali, Ophir Münz-Manor, Benny Kimelfeld, and Yonatan Belinkov.

The dataset is available in the folder "data/prepared_data". It's already split into train, dev, and test sets.
You can also find the raw data in the folder "data/raw_data".
For convenience, we also uploaded a version of the dataset to huggingface's datasets repository: 
https://huggingface.co/datasets/tokeron/Piyyut

This version splitted into the common form of annotated datasets in metaphor detection:
Each example is a sentence, a word, and it's label as metaphorical or literal in the context of the sentence.

To reproduce the results in the paper, you can use the following scripts:
- To train (fine-tune) a model (AlephBERT or BEREL) on one of the 'Piyyut' datasets (Piyyut or Pinchas)
  run the script "train_classification.py" with the following arguments:
  --model_type: The model type to use. Can be 'aleph_bert' or 'berel'.
  --dataset_name: The dataset to use. Can be 'pre_piyyut', 'Pinchas' or 'all' (for the combined dataset).
- To train a model on the mlm task with Ben-Yehuda corpus:
  run the script "train_mlm.py".
- To prepare the dataset for the mlm task:
  run the script "data/prepare_data_mlm.py".
- To prepare the dataset for the classification task:
  run the script "data/prepare_data_fl.py".


Paper Abstract: The corpora of late antique and medieval Hebrew texts are vast. They represent a crucial linguistic and cultural bridge between Biblical and modern Hebrew. 
Poetry is prominent in these corpora and one of its main characteristics is the frequent use of metaphor. Distinguishing figurative and literal language use is a major task for scholars of the Humanities, especially in the fields of literature, linguistics, and hermeneutics.
This paper presents a new, challenging dataset of late antique and medieval Hebrew poetry with expert annotations of metaphor, as well as some baseline results, which we hope will facilitate further research in this area.