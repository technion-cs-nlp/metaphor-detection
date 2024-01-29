import wandb
from transformers import AutoModelForTokenClassification, BertTokenizerFast, TrainingArguments, EarlyStoppingCallback
import datasets
from transformers import DataCollatorForTokenClassification, TrainerCallback, DataCollatorWithPadding, \
    BertForSequenceClassification, BertConfig
from helpers.utils import *
from config.config_parser import *
from trainers.wce_trainer import WeightedLossTrainer
import copy
import transformers
from transformers import  DefaultDataCollator
import argparse
from transformers import BertModel

def f1_objective(metrics):
    return metrics['eval_f1']

def set_seed_for_all(seed):
    # Set seed for reproducibility
    set_seed(seed)
    transformers.set_seed(seed)

def load_datasets(args):
    dataset_name = args.dataset_name
    corpus = args.corpus
    # TODO move to helper function in utils

    raw_datasets = datasets.DatasetDict({datasets.Split.TRAIN: load_dataset_split(corpus, "train"),
                                             datasets.Split.VALIDATION: load_dataset_split(corpus, "validation"),
                                             datasets.Split.TEST: load_dataset_split(corpus, "test")})
    return raw_datasets

def load_tokenizer_and_tokenize(args, raw_datasets):
    dataset_name = args.dataset_name
    ignore_subtokens = training_args.data_args.ignore_subtokens
    berel = training_args.paths.berel_path

    model_type = args.model_type
    model_checkpoint = args.checkpoint
    dataset_method = args.lm
    additional_layers = args.add_l
    # Load tokenizer
    tokenizer, tokenized_datasets = load_tokenizer(model_checkpoint, model_type, berel_path,
                                                   raw_datasets, ignore_subtokens, dataset_method, dataset_name)
    return {
        "tokenizer": tokenizer,
        "tokenized_datasets": tokenized_datasets
        }


def load_trainer(args, experiment_name, tokenizer, tokenized_datasets, wandb_init=True):
    report_train = training_args.model_args.report_train
    logging_steps = training_args.model_args.logging_steps
    do_train = training_args.model_args.do_train
    do_eval = training_args.model_args.do_eval
    do_predict = training_args.model_args.do_predict
    eval_steps = training_args.model_args.eval_steps
    save_steps = training_args.model_args.save_steps
    save_total_limit = training_args.model_args.save_total_limit
    save_strategy = training_args.model_args.save_strategy
    load_best_model_at_end = training_args.model_args.load_best_model_at_end
    evaluation_strategy = training_args.model_args.evaluation_strategy
    logging_strategy = training_args.model_args.logging_strategy
    main_dir = training_args.paths.main_dir
    data_dir = os.path.join(main_dir, training_args.paths.data_dir)
    berel_path = training_args.paths.berel_path

    # data args from config
    rows_per_example = training_args.data_args.rows_per_example
    ignore_subtokens = training_args.data_args.ignore_subtokens
    # args from main
    wandb_project = args.wandb_name
    model_type = args.model_type
    model_checkpoint = args.checkpoint
    lr = args.lr
    per_device_train_batch_size = args.bs
    epochs = args.ep
    weight_decay = args.w_decay
    weighted_loss = args.w_loss
    metaphor_weight = args.metaphor_w
    non_metaphor_weight = args.nmetaphor_w
    gradient_accumulation_steps = args.gas
    random_weights = args.random_w
    dataset_method = args.lm
    warmup_steps = args.warmup_s
    warmup_ratio = args.warmup_r
    lr_scheduler_type = args.lr_sched
    seed = args.seed
    corpus = args.corpus
    esp = args.esp
    dataset_name = args.dataset_name

    model_checkpoint = get_model_checkpoint(model_type, model_checkpoint, dataset_name)
    # dataset, dataset_name = get_dataset_name(model_type, dataset)

    # Set seed for reproducibility
    set_seed(seed)
    transformers.set_seed(seed)
    # initialize_wandb
    # TODO move to helper function in utils
    wandb_config = {
        "model_type": model_type,
        "model_name_or_path": model_checkpoint,
        "per_device_train_batch_size": per_device_train_batch_size,
        "num_epochs": epochs,
        "learning_rate": lr,
        "weighted_loss": weighted_loss,
        "non_metaphor_weight": non_metaphor_weight,
        "metaphor_weight": metaphor_weight,
        "model_checkpoint": model_checkpoint,
        "ignore_subtokens": ignore_subtokens,
        "layer_index_for_classification": additional_layers if len(additional_layers) > 0 else "",
        "dataset_name": corpus, "experiment_name": experiment_name
    }

    if wandb_init:
        try:
            wandb.init(project=wandb_project, entity=training_args.wandb.entity,
                       name=experiment_name, config=wandb_config)
        except:
            wandb.login(key=training_args.wandb.key)
            wandb.init(project=wandb_project, entity=training_args.wandb.entity,
                       name=experiment_name, config=wandb_config)
    # Load dataset



    # path_words_statistics = generate_word_statistics(dataset_method, model_type, experiment_name, raw_datasets)
    # TODO move to models file/dir /Utils
    def bert_init():
        if model_type == 'aleph_bert':
            label_names = ["O", "B-metaphor", "I-metaphor"]
            id2label = {str(i): label for i, label in enumerate(label_names)}
            label2id = {v: k for k, v in id2label.items()}
            if 'alephbertgimmel' in model_checkpoint:
                config = BertConfig.from_pretrained(model_checkpoint,
                                                    output_hidden_states=use_more_layers,
                                                    id2label=id2label,
                                                    label2id=label2id)
            elif 'xlm' in model_checkpoint:
                config = BertConfig.from_pretrained("xlm-roberta-base",
                                                    output_hidden_states=use_more_layers,
                                                    id2label=id2label,
                                                    label2id=label2id)
            else:
                config = BertConfig.from_pretrained("onlplab/alephbert-base",
                                                    output_hidden_states=use_more_layers,
                                                    id2label=id2label,
                                                    label2id=label2id)
            # Load a model with custom number of layers in the classification head
            if use_more_layers and additional_layers_num > 0:
                model = AlephBertWideHead(layers_for_cls=additional_layers,
                                          only_intermid_rep=only_intermediate_representation)
            else:
                # Load pretrained model with a token classification head
                model = AutoModelForTokenClassification.from_pretrained(
                    model_checkpoint,
                    config=config,
                )
        elif model_type == 'simple_melbert':
            model = alephMelBert(layers_for_cls=additional_layers, only_intermid_rep=only_intermediate_representation)
        elif model_type == 'per_word':
            if dataset_name == 'MOH':
                model_name_to_load = 'bert-base-uncased'
            else:  # MHeb
                model_name_to_load = 'onlplab/alephbert-base'
            label_names = ["O", "metaphor"]
            id2label = {str(i): label for i, label in enumerate(label_names)}
            label2id = {v: k for k, v in id2label.items()}
            config = BertConfig.from_pretrained(model_name_to_load,
                                                output_hidden_states=use_more_layers,
                                                id2label=id2label,
                                                label2id=label2id)
            model = BertForSequenceClassification.from_pretrained(model_name_to_load, config=config)
            # model = AlephBertPerWord(layers_for_cls=additional_layers, only_intermid_rep=only_intermediate_representation)
        elif model_type == 'mT5':
            label_names = ["O", "B-metaphor", "I-metaphor"]
            id2label = {str(i): label for i, label in enumerate(label_names)}
            label2id = {v: k for k, v in id2label.items()}
            config = BertConfig.from_pretrained(model_checkpoint,
                                                output_hidden_states=use_more_layers,
                                                id2label=id2label,
                                                label2id=label2id)
            model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=config)
        elif model_type == 'berel':
            label_names = ["O", "B-metaphor", "I-metaphor"]
            id2label = {str(i): label for i, label in enumerate(label_names)}
            label2id = {v: k for k, v in id2label.items()}
            config = BertConfig.from_pretrained(berel_path,
                                                output_hidden_states=use_more_layers,
                                                id2label=id2label,
                                                label2id=label2id)
            model = AutoModelForTokenClassification.from_pretrained(
                berel_path,
                config=config,
            )
        else:
            raise ValueError('Unknown model type')

        # initialize model weights with random weights
        if random_weights:
            model.init_weights()
        return model

    if model_type == 'mT5':
        data_collator = DataCollatorForSeq2Seq(tokenizer)
        mt5_args = Seq2SeqTrainingArguments(
            experiment_name,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            logging_strategy=evaluation_strategy,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            learning_rate=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            weight_decay=weight_decay,
            save_total_limit=save_total_limit,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            report_to="wandb",
            include_inputs_for_metrics=True,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
        )

        # Function that returns an untrained model to be trained
        def model_init_mt5():
            return MT5ForConditionalGeneration.from_pretrained(model_checkpoint)

        def compute_metrics_mt5(eval_pred):
            bad_predictions = None
            inputs = eval_pred.inputs
            labels = eval_pred.label_ids
            predictions = eval_pred.predictions
            tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            metric = load_metric("seqeval")
            label_names = ["O", "B-metaphor", "I-metaphor"]
            if dataset_method == 'interlaced':
                true_labels = [[l for l in label if l != -100] for label in labels]
                decoded_labels = tokenizer.batch_decode(true_labels, skip_special_tokens=True)
                decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
                # full_df = pd.DataFrame({"labels": decoded_labels, "predictions": decoded_preds})
                interlaced_true_labels = []
                interlaced_true_predictions = []
                for label_array in decoded_labels:
                    interlaced_current = []
                    for i in range(len(label_array)):
                        if str(label_array[i:]).startswith('B-metaphor'):
                            interlaced_current.append('B-metaphor')
                            i += len('B-metaphor')
                        elif str(label_array[i:]).startswith('I-metaphor'):
                            interlaced_current.append('I-metaphor')
                            i += len('I-metaphor')
                        elif str(label_array[i:]).startswith('O'):
                            interlaced_current.append('O')
                            i += len('O')
                    interlaced_true_labels.append(interlaced_current)

                for prediction_array in decoded_preds:
                    interlaced_current = []
                    for i in range(len(prediction_array)):
                        if str(prediction_array[i:]).startswith('B-metaphor'):
                            interlaced_current.append('B-metaphor')
                            i += len('B-metaphor')
                        elif str(prediction_array[i:]).startswith('I-metaphor'):
                            interlaced_current.append('I-metaphor')
                            i += len('I-metaphor')
                        elif str(prediction_array[i:]).startswith('O'):
                            interlaced_current.append('O')
                            i += len('O')
                    interlaced_true_predictions.append(interlaced_current)

                true_predictions = []
                true_preds = interlaced_true_predictions
                true_labels = interlaced_true_labels
                count = 0
                bad_count = 0
                full_df = pd.DataFrame({"labels": true_labels, "predictions": true_preds})
                for label_array, pred_array, orig_pred, orig_labels, input in \
                        zip(true_labels, true_preds, decoded_preds, decoded_labels, decoded_inputs):
                    count += 1
                    if len(label_array) != len(pred_array):
                        print(f'Original prediction: {orig_pred}')
                        print(f'Original labels: {orig_labels}')
                        print(f'Input: {input}')

                        bad_count += 1
                        pred_array = [label_names[0] for _ in range(len(label_array))]
                        true_predictions.append(pred_array)
                    else:
                        true_predictions.append(pred_array)
                        print(f'true prediction: {pred_array}')
                        print(f'true labels: {label_array}')
                bad_predictions = float(bad_count / count)
            elif dataset_method == 'tag':
                true_labels = [[l for l in label if l != -100] for label in labels]
                decoded_labels = tokenizer.batch_decode(true_labels, skip_special_tokens=True)
                true_labels = [[label_names[1] if l == 497 else label_names[0] for l in label] for label in true_labels]
                true_predictions = []
                count = 0
                bad_count = 0
                for label_array, pred_array in zip(true_labels, decoded_preds):
                    count += 1
                    pred_array = pred_array.split()
                    pred_array = [label_names[int(p)] for p in pred_array if p == '0' or p == '1']
                    if len(label_array) != len(pred_array):
                        bad_count += 1
                        pred_array = [label_names[0] for _ in range(len(label_array))]
                        true_predictions.append(pred_array)
                    else:
                        true_predictions.append(pred_array)
                bad_predictions = float(bad_count / count)
            elif dataset_method == 'natural':
                inputs = [[i for i in input if i != -100] for input in inputs]
                decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
                input_sentence_split = [decoded_input.replace('. המילים הבאות הן מטאפורות: <extra_id_0>', '').split()
                                  for decoded_input in decoded_inputs]
                true_labels = [[l for l in label if l != -100] for label in labels]
                decoded_labels = tokenizer.batch_decode(true_labels, skip_special_tokens=True)
                metaphor_words_labels = [decoded_labels[i].split('<extra_id_0>:')[-1].split(",")
                                         for i in range(len(decoded_labels))]
                # clean empty strings from the list
                metaphor_words_labels = [[x for x in l if x != ' '] for l in metaphor_words_labels]
                metaphor_words_predictions = [decoded_preds[i].split('<extra_id_0>:')[-1].split(",") for i in
                                              range(len(decoded_preds))]

                df = pd.DataFrame({'metaphor_words_labels': metaphor_words_labels,
                                   'metaphor_words_predictions': metaphor_words_predictions})

                TP, TN, FP, FN = 0, 0, 0, 0
                for i, (input_sentence, metaphor_list, prediction_list) in \
                        enumerate(zip(input_sentence_split, metaphor_words_labels, metaphor_words_predictions)):
                    for word in input_sentence:
                        if word in metaphor_list:
                            if word in prediction_list:
                                TP += 1
                            else:
                                FN += 1
                        else:
                            if word in prediction_list:
                                FP += 1
                            else:
                                TN += 1

                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0


                return {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                }
                    # "sklearn_precision": 0,
                    # "sklearn_recall": 0,
                    # "sklearn_f1": 0,
                    # "sklearn_f1_macro": 0,
                    # "sklearn_f1_weighted": 0,
                    # "sklearn_accuracy": 0,
                # }

            # elif dataset_method == 'natural_per_word':
            #     true_labels = [[l for l in label if l != -100] for label in labels]
            #     decoded_labels = tokenizer.batch_decode(true_labels, skip_special_tokens=True)
            #     true_labels = [label_names[1] if (decoded_labels[i].split('<extra_id_0>:')[-1] == 'מטאפורה')
            #                    else label_names[0] for i in range(len(decoded_labels))]
            #
            #     true_predictions = [label_names[1] if (decoded_preds[i].split('<extra_id_0>:')[-1] == 'מטאפורה')
            #                         else label_names[0] for i in range(len(decoded_preds))]
            #     print("Percentage of 'O' predictions: {}".format((true_predictions.count('O') / len(true_predictions))))
            #     wandb.log({"bad_predictions": (true_predictions.count('O') / len(true_predictions))})
            #     all_metrics = metric.compute(predictions=[true_predictions], references=[true_labels])
            else:
                raise ValueError("Invalid method name")

            if dataset_method != 'natural_per_word':
                all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
                true_labels = [label for sublist in true_labels for label in sublist]
                true_predictions = [label for sublist in true_predictions for label in sublist]
            else:
                raise ValueError("Implement this")

            binary_labels = [1 if (l == "B-metaphor" or l == "I-metaphor") else 0 for l in true_labels]
            binary_predictions = [1 if (p == "B-metaphor" or p == "I-metaphor") else 0 for p in true_predictions]

            results = {
                "precision": all_metrics["overall_precision"],
                "recall": all_metrics["overall_recall"],
                "f1": all_metrics["overall_f1"],
                "accuracy": all_metrics["overall_accuracy"],
                "sklearn_precision": metrics.precision_score(binary_labels, binary_predictions),
                "sklearn_recall": metrics.recall_score(binary_labels, binary_predictions),
                "sklearn_f1": metrics.f1_score(binary_labels, binary_predictions),
                # "sklearn_f1_macro": metrics.f1_score(binary_labels, binary_predictions, average="macro"),
                # "sklearn_f1_weighted": metrics.f1_score(binary_labels, binary_predictions, average="weighted"),
                "sklearn_accuracy": metrics.accuracy_score(binary_labels, binary_predictions),
                "%good_predictions": 1 - bad_predictions if bad_predictions is not None else -1
            }
            return results

        trainer = Seq2SeqTrainer(
            model_init=model_init_mt5,
            args=mt5_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            # eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_mt5,
        )
    else:
        # Putting together samples inside a batch
        if model_type == 'per_word':
            data_collator = DefaultDataCollator()
            # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
        else:
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Initialize training arguments for trainer
        bert_args = TrainingArguments(
            experiment_name,
            evaluation_strategy=evaluation_strategy,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            learning_rate=lr,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            run_name=experiment_name,
            report_to=["wandb"],
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model='f1',
            save_total_limit=save_total_limit,
            gradient_accumulation_steps=gradient_accumulation_steps,
            include_inputs_for_metrics=True,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
        )

        melbert, compute_metrics, per_example_label = get_model_specific_args(model_type)

        callbacks = []

        if save_strategy != "no":
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=esp))

        trainer = WeightedLossTrainer(
            args=bert_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            use_more_layers=use_more_layers,
            layers_for_classification=additional_layers,
            only_intermediate_representation=only_intermediate_representation,
            per_example_label=per_example_label,
            non_metaphor_weight=non_metaphor_weight,
            metaphor_weight=metaphor_weight,
            melbert=melbert,
            model_init=bert_init,
            callbacks=callbacks,
            dataset_name=dataset_name,
        )

    if report_train:
        class CustomCallback(TrainerCallback):
            def __init__(self, trainer) -> None:
                super().__init__()
                self._trainer = trainer

            def on_epoch_end(self, args, state, control, **kwargs):
                if control.should_evaluate:
                    control_copy = copy.deepcopy(control)
                    self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                    return control_copy

        trainer.add_callback(CustomCallback(trainer))
    print('Training loaded')

    return trainer


def prepare_hyperparameter(trainer, experiment_name):
    best_trial = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space,
                                               compute_objective=f1_objective, n_trials=n_trials)

    wandb.log({'Experiment': experiment_name,
               'objective': best_trial.objective,
               'hyperparameters': best_trial.hyperparameters,
               'run_id': best_trial.run_id})

    table = wandb.Table(columns=["run_id", "objective", "hyperparameters"])
    table.add_data(best_trial.run_id, best_trial.objective, best_trial.hyperparameters)
    wandb.log({"Best Model": table})
    # Print the best trial hyperparameters and objective values into a txt file
    with open("best_trial.txt", "w") as f:
        f.write("Best trial hyperparameters: " + str(best_trial.hyperparameters) + "\n")
        f.write("Best trial objective: " + str(best_trial.objective) + "\n")
        f.write("Best trial run_id: " + str(best_trial.run_id) + "\n")
