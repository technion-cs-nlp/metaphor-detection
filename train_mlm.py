import os

import wandb
from transformers import BertTokenizerFast, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForMaskedLM,\
    DataCollatorForLanguageModeling
from helpers.utils import *
import pandas as pd
import collections
import numpy as np
from transformers import default_data_collator
import yaml
from box import Box
from tqdm import tqdm
import argparse
from config.config_parser import training_args


def train_mlm(args):
    num_train_epochs = args.epochs
    lr = args.lr
    ordered = args.ordered
    model_checkpoint = args.model_checkpoint
    batch_size = args.bs
    warmup_steps = args.warmup_steps
    warmup_ratio = args.warmup_ratio
    lr_scheduler_type = args.lr_scheduler_type
    model_name = 'bert_'
    experiment_name = '{}mlm_ep_{}_lr_{}_{}_all_data'.format(model_name, num_train_epochs, lr,
                                                    'ordered' if ordered else 'unordered')
    wandb_config = {
        "num_train_epochs": num_train_epochs,
        "lr": lr, "ordered": ordered,
        "model_checkpoint": model_checkpoint,
        "batch_size": batch_size
    }

    try:
        entity = training_args.wandb.entity
    except:
        entity = None

    if entity is not None:
        wandb.init(project="mlm_training", entity=entity,
                       name=experiment_name, config=wandb_config)
        print('Experiment name: {}'.format(experiment_name))

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    chunk_size = 128
    # tokenizer = AutoTokenizer.from_pretrained(training_args.model_args.model_checkpoint)
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
    pseudocatalogue_folder = 'data/public_domain_dump'
    pseudocatalogue = 'pseudocatalogue.csv'
    path_to_catalog = os.path.join(pseudocatalogue_folder, pseudocatalogue)
    # load pseudocatalogue from csv
    catalog_df = pd.read_csv(path_to_catalog)
    # get all unique pseudocatalogue entries in genre
    genres = catalog_df['genre'].unique()

    mlm_data_path = 'data/mlm_data'

    genres_list = [
        # 'tanach',
        # 'mishna',
        # 'tosefta',
        # 'yerushalmi',
        # 'midrashtanchuma',
        # 'midrashtehilim',
        # 'midrashraba',
        'search',
        'dictionaries&lexicons',
        'articles&essays',
        'letters',
        'diaries',
        'prose',
        'plays',
        'parables',
        'poetry',
        'yose',
        'piyyut',
        'pinechas'
        ]

    full_path = ['{}/{}.csv'.format(mlm_data_path, corpora) for corpora in genres_list]

    # calculate statistics
    # read all csv files into one dataframe
    df = pd.concat([pd.read_csv(f) for f in full_path], ignore_index=True)
    # count the total number of rows in the dataframe
    total_rows = df.shape[0]
    # get the total number of words in the dataframe
    total_words = df['text'].str.split().str.len().sum()
    print('Total rows: {}'.format(total_rows))
    print('Total words: {}'.format(total_words))

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            for i in range(len(result["input_ids"])):
                result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    # We can build our own data_collator to mask whole tokens with [MASK] token
    def whole_word_masking_data_collator(features):
        wwm_probability = 0.2
        for feature in features:
            word_ids = feature.pop("word_ids")
            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)
            # Randomly mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
        return default_data_collator(features)

    # full_path = []
    # for genre in genres_list:
    #     full_path.append(os.path.join(data_path, genre + '.csv'))

    checkpoint_name = experiment_name

    # Show the training loss with every epoch
    logging_steps = 4
    model_name = model_checkpoint.split("/")[-1]
    # This data_collator can mask sub-tokens with [MASK] token instead of masking the whole token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.2)
    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned_fbt_epochs_from_unsupervised",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=logging_steps,
        learning_rate=lr,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=True,
        save_total_limit=1,
        num_train_epochs=num_train_epochs,
        report_to="wandb",
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type
    )

    for path in full_path:
        if not os.path.exists(path):
            raise ValueError('Path {} does not exist'.format(path))

    unsupervised_dataset = load_dataset('csv', data_files=full_path)

    # shuffle
    if not ordered:
        unsupervised_dataset = unsupervised_dataset.shuffle(seed=42)

    validation_split_size = 0.01  # 0.01 = 1% of the data will be used for validation

    unsupervised_dataset = unsupervised_dataset['train'].train_test_split(validation_split_size)

    unsupervised_train_dataset = unsupervised_dataset['train']
    unsupervised_validation_dataset = unsupervised_dataset['test']

    tokenized_train_datasets = unsupervised_train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    tokenized_validation_datasets = unsupervised_validation_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Group texts into chunks of max_len
    train_dataset = tokenized_train_datasets.map(group_texts, batched=True)
    eval_dataset = tokenized_validation_datasets.map(group_texts, batched=True)

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,  # whole_word_masking_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    print("Finished training on all genres")
    # Save the model
    out_folder = 'results/model_after_pretraining_2'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    trainer.save_model('{}/{}'.format(out_folder, experiment_name))
    print("Saved model to {}".format(out_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='onlplab/alephbert-base')
    parser.add_argument('--lr', type=float, default=1e-4) # 2.0e-05
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--ordered', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default='data/unsupervised')
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')


    args = parser.parse_args()
    train_mlm(args)
