import wandb
from helpers.utils import *
from config.config_parser import *
import argparse
from load_trainer import *
from transformers import hf_argparser

# def f1_objective(metrics):
#     return metrics['eval_f1']


# main function
def train_classification(args):
    # # for debugging
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    n_trials = args.n_trials
    hyperparameter_search = args.hyperparameter_search
    model_type = args.model_type
    corpus = args.corpus
    save_model = training_args.model_args.save_model
    save_torch_model = training_args.model_args.save_torch_model
    model_checkpoint = args.checkpoint
    save_strategy = training_args.model_args.save_strategy
    main_dir = training_args.paths.main_dir
    predictions_dir = os.path.join(main_dir, training_args.paths.predictions_dir)
    ignore_subtokens = training_args.model_args.ignore_subtokens

    # TODO replace print with log to wandb
    print("Starting training of classification model")

    set_seed_for_all(args.seed)
    raw_datasets = load_datasets(args)
    tokenizer_res = load_tokenizer_and_tokenize(args, raw_datasets=raw_datasets)
    tokenizer = tokenizer_res['tokenizer']
    tokenized_datasets = tokenizer_res['tokenized_datasets']
    experiment_name = get_experiment_name(args)
    trainer = load_trainer(args,
                           experiment_name=experiment_name,
                           tokenizer=tokenizer,
                           tokenized_datasets=tokenized_datasets)


    if hyperparameter_search:
        prepare_hyperparameter(trainer=trainer, experiment_name=experiment_name)

    else:
        print("Start training...")
        trainer.train()
        print("Finished training")

        # Save the model
        main_dir = training_args.paths.main_dir
        checkpoints_dir = os.path.join(main_dir, training_args.paths.results_checkpoints_dir)
        # create a directory for each experiment type
        after_pretraining = "dapt_" if 'after_pretraining' in model_checkpoint else ""
        experiment_type_dir = '{}_{}'.format(model_type, after_pretraining)
        checkpoints_dir = os.path.join(experiment_type_dir, checkpoints_dir)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if save_model:
            trainer.save_model(os.path.join(checkpoints_dir, experiment_name))
        test_all_ancient_hebrew_flag = False
        if test_all_ancient_hebrew_flag:
            test_all_ancient_hebrew(trainer, tokenizer, ignore_subtokens, wandb, experiment_name)

        save_predictions_on_all_sets_flag = True
        save_predictions_on_all_sets(trainer, tokenizer, save_model, save_strategy,
                                 tokenized_datasets, raw_datasets, wandb, experiment_name, predictions_dir)


    print("Done everything")


if __name__ == '__main__':
    # TODO: Use HF parser
    # TODO move to another file
    # TODO make subparsers for each argmument type
    parser = argparse.ArgumentParser(description='Train a classification model')

    # basic setup
    parser.add_argument('--wandb_name', type=str, default='fine_tuning_fl', help='Name of the experiment in wandb')
    parser.add_argument('--model_type', type=str, default='aleph_bert', help='Model type',
                        choices=['aleph_bert', 'berel'])
    parser.add_argument('--checkpoint', type=str, default='bert-base-uncased', help='Model checkpoint')
    parser.add_argument('--lr', type=float, default=0.000054, help='Learning rate')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--ep', type=int, default=8, help='Number of epochs')
    parser.add_argument('--w_decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--gas', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--random_w', type=bool, default=False, help='Random weights')
    parser.add_argument('--warmup_s', type=int, default=100, help='Warmup steps')  # 400
    parser.add_argument('--warmup_r', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--lr_sched', type=str, default='linear', help='Learning rate scheduler type',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                 'constant_with_warmup'])
    # Weghted loss
    parser.add_argument('--w_loss', type=bool, default=True, help='Use weighted loss')
    parser.add_argument('--metaphor_w', type=float, default=9.0, help='Weight of metaphor loss') # 9.0
    parser.add_argument('--nmetaphor_w', type=float, default=1.0, help='Weight of non-metaphor loss')

    # hyperparameter search
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials for hyperparameter search')
    parser.add_argument('--hyperparameter_search', type=bool, default=False, help='Hyperparameter search')

    # mT5
    parser.add_argument('--lm', type=str, default='interlaced', help='Labeling method',
                        choices=['interlaced', 'tag', 'natural'])

    # intermediate representation
    parser.add_argument('--only_ir', type=bool, default=False, help='Only use intermediate representation')
    parser.add_argument('--use_ml', type=bool, default=False, help='Use more layers')
    parser.add_argument('--add_l', type=str, default='3_4', help='Additional layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--corpus', type=str, default='pre_piyut_1', help='Corpus',
                        choices=['pre_piyut_1', 'pinchas_1', 'all_1'])
    parser.add_argument('--esp', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--dataset_name', type=str, default='pre_piyut', help='Dataset',
                        choices=['pre_piyut', 'pinchas', 'all'])

    args = parser.parse_args()
    train_classification(args)
