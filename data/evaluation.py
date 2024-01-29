import numpy as np
import pandas as pd
import os
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import argparse
from bidi.algorithm import get_display

def create_confusion_matrix(cf_matrix, title, filename, plots_dir):
    """
    :param cf_matrix: confusion matrix
    :param cf_matrix:
    :return:
    """
    ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='viridis')

    ax.set_title(title)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    ax.xaxis.set_ticklabels(['Not Metaphor', 'Metaphor'])
    ax.yaxis.set_ticklabels(['Not Metaphor', 'Metaphor'])

    # tight layout fixes the figure size
    plt.tight_layout()
    plt.savefig('{}/{}'.format(plots_dir, filename))
    plt.show()


def evaluate_results(predictions_path, plots_dir, experiment_name):
    """
    Evaluate the predictions of the model.
    :param predictions_path: The path to the predictions file.
    :param plots_dir: The directory where the plots are saved.
    """
    try:
        predictions = pd.read_pickle(predictions_path)
    except:
        predictions = pd.read_json(predictions_path.replace('.pkl', '.json'))
    # drop the 'data' column
    predictions = predictions.drop(columns=['data'])

    # join the list of labels (concatenate all the labels)
    labels = np.concatenate(predictions['labels'])

    # join the list of predictions (concatenate all the labels)
    predictions = np.concatenate(predictions['predictions'])

    # calculate the confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(labels, predictions)

    # plot the confusion matrix (True is a metaphor, False is a non-metaphor)
    # disp.plot(include_values=True)

    plt.title('Confusion matrix - {}'.format(experiment_name))
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.yticks(rotation=90)
    plt.xticks([0, 1], ['Not Metaphor', 'Metaphor'])
    plt.yticks([0, 1], ['Not Metaphor', 'Metaphor'])
    plt.tight_layout()

    # save the plot (if the plots_dir doesn't exist, create it)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, '{}_confusion_matrix.png'.format(experiment_name)))
    plt.show()
    # save classification report into file
    class_report = classification_report(labels, predictions)
    with open(os.path.join(plots_dir, '{}_classification_report.txt'.format(experiment_name)), 'w') as f:
        f.write(class_report)
    print(class_report)


def calculate_words_statistics(raw_dataset):
    metaphor_words = []
    non_metaphor_words = []
    words_statistics = {}
    # iterate over the dadaframe
    for index, (data, labels) in enumerate(zip(raw_dataset['sentence'], raw_dataset['label'])):
        # iterate over the data and labels
        for word, label in zip(data, labels):
            if word not in words_statistics:
                if label == 0:
                    words_statistics[word] = {'metaphor': 0, 'non_metaphor': 1}
                    non_metaphor_words.append(word)
                else:  # It's a metaphor
                    words_statistics[word] = {'metaphor': 1, 'non_metaphor': 0}
                    metaphor_words.append(word)
            else:
                if label == 0:
                    words_statistics[word]['non_metaphor'] += 1
                    non_metaphor_words.append(word)
                else:  # It's a metaphor
                    words_statistics[word]['metaphor'] += 1
                    metaphor_words.append(word)
    return words_statistics, metaphor_words, non_metaphor_words


def full_eval(data, labels, predictions, metaphor_words, plots_dir, experiment_name, non_metaphor_words=None):
    mistake_dict_FP = {}
    mistake_dict_FN = {}
    dict_TP = {}
    dict_TN = {}
    unseen_TP, unseen_FP, unseen_FN, unseen_TN = "", "", "", ""
    seen_TP, seen_FP, seen_FN, seen_TN = "", "", "", ""
    total_seen, total_unseen = 0, 0
    unseen_TP_count, unseen_FP_count, unseen_FN_count, unseen_TN_count = 0, 0, 0, 0
    seen_TP_count, seen_FP_count, seen_FN_count, seen_TN_count = 0, 0, 0, 0
    for data_index, (data_item, labels_item, predictions_item) in enumerate(zip(data, labels, predictions)):
        for word_index, (word, label, prediction) in enumerate(zip(data_item, labels_item, predictions_item)):
            if label != 0 and prediction == 0:
                if word in metaphor_words:
                    seen_FN += " " + word + " "
                    seen_FN_count += 1
                    total_seen += 1
                else:
                    unseen_FN += " " + word + " "
                    unseen_FN_count += 1
                    total_unseen += 1
                if word not in mistake_dict_FN:
                    mistake_dict_FN[word] = 1
                else:
                    mistake_dict_FN[word] += 1
            elif label == 0 and prediction != 0:
                if word in metaphor_words:
                    seen_FP += " " + word + " "
                    seen_FP_count += 1
                    total_seen += 1
                else:
                    unseen_FP += " " + word + " "
                    unseen_FP_count += 1
                    total_unseen += 1
                if word not in mistake_dict_FP:
                    mistake_dict_FP[word] = 1
                else:
                    mistake_dict_FP[word] += 1

            elif label == 0 and prediction == 0:
                if word in metaphor_words:
                    seen_TN += " " + word + " "
                    seen_TN_count += 1
                    total_seen += 1
                else:
                    unseen_TN += " " + word + " "
                    unseen_TN_count += 1
                    total_unseen += 1
                if word not in dict_TN:
                    dict_TN[word] = 1
                else:
                    dict_TN[word] += 1
            elif label != 0 and prediction != 0:
                if word in metaphor_words:
                    seen_TP += " " + word + " "
                    seen_TP_count += 1
                    total_seen += 1
                else:
                    unseen_TP += " " + word + " "
                    unseen_TP_count += 1
                    total_unseen += 1
                if word not in dict_TP:
                    dict_TP[word] = 1
                else:
                    dict_TP[word] += 1
    # get the 10 most common mistakes
    mistake_dict_FP = {k: v for k, v in sorted(mistake_dict_FP.items(), key=lambda item: item[1], reverse=True)}
    mistake_dict_FN = {k: v for k, v in sorted(mistake_dict_FN.items(), key=lambda item: item[1], reverse=True)}
    dict_TP = {k: v for k, v in sorted(dict_TP.items(), key=lambda item: item[1], reverse=True)}
    dict_TN = {k: v for k, v in sorted(dict_TN.items(), key=lambda item: item[1], reverse=True)}
    # print the 10 most common mistakes
    print("non_metaphor_words", len(non_metaphor_words))
    print("metaphor_words", len(metaphor_words))
    print("The x most common mistakes (FP):")
    x = 10
    for key, value in list(mistake_dict_FP.items())[:x]:
        # Check if the word appears in the non-metaphor words
        is_in_non_metaphor_list = non_metaphor_words.count(key)
        is_in_metaphor_list = metaphor_words.count(key)
        print(get_display(key), value, ". # train as non metaphor: " + str(is_in_non_metaphor_list),
              ". # train as metaphor: " + str(is_in_metaphor_list))
    print("The 10 most common mistakes (FN):")
    for key, value in list(mistake_dict_FN.items())[:x]:
        # Check if the word appears in the non-metaphor words
        is_in_non_metaphor_list = non_metaphor_words.count(key)
        is_in_metaphor_list = metaphor_words.count(key)
        print(get_display(key), value, ". # train as non metaphor: " + str(is_in_non_metaphor_list),
                ". #  train as metaphor: " + str(is_in_metaphor_list))
    print("The 10 most common true positives:")
    for key, value in list(dict_TP.items())[:x]:
        # Check if the word appears in the non-metaphor words
        is_in_non_metaphor_list = non_metaphor_words.count(key)
        is_in_metaphor_list = metaphor_words.count(key)
        print(get_display(key), value, "is in train as non metaphor: " + str(is_in_non_metaphor_list),
                "is in train as metaphor: " + str(is_in_metaphor_list))
    print("The 10 most common true negatives:")
    for key, value in list(dict_TN.items())[:x]:
        # Check if the word appears in the non-metaphor words
        is_in_non_metaphor_list = non_metaphor_words.count(key)
        is_in_metaphor_list = metaphor_words.count(key)
        print(get_display(key), value, ". # train as non metaphor: " + str(is_in_non_metaphor_list),
                ". #  train as metaphor: " + str(is_in_metaphor_list))

    # investigate the word 'מים'
    print("dict_TN: ", dict_TN['מים'])
    print("dict_TP:", dict_TP['מים'])
    print("mistake_dict_FP:", mistake_dict_FP['מים'])
    print("mistake_dict_FN:", mistake_dict_FN['מים'])

    # investigate the word 'על'
    print("dict_TN: ", dict_TN['על'])
    print("dict_TP:", dict_TP['על'])
    print("mistake_dict_FP:", mistake_dict_FP['על'])
    print("mistake_dict_FN:", mistake_dict_FN['על'])




    # create confusion matrix for all data
    total_TP_count = seen_TP_count + unseen_TP_count
    total_FP_count = seen_FP_count + unseen_FP_count
    total_FN_count = seen_FN_count + unseen_FN_count
    total_TN_count = seen_TN_count + unseen_TN_count
    cf_matrix_total = np.array([[total_TN_count, total_FP_count], [total_FN_count, total_TP_count]])
    cf_matrix_seen = np.array([[seen_TN_count, seen_FP_count], [seen_FN_count, seen_TP_count]])
    cf_matrix_unseen = np.array([[unseen_TN_count, unseen_FP_count], [unseen_FN_count, unseen_TP_count]])

    # Calculate accuracy, precision, recall, f1-score of seen words
    if total_seen == 0:
        accuracy_seen = 0
    else:
        accuracy_seen = (seen_TP_count + seen_TN_count) / total_seen
    if seen_TP_count + seen_FP_count == 0:
        precision_seen = 0
    else:
        precision_seen = seen_TP_count / (seen_TP_count + seen_FP_count)
    if seen_TP_count + seen_FN_count == 0:
        recall_seen = 0
    else:
        recall_seen = seen_TP_count / (seen_TP_count + seen_FN_count)
    if precision_seen + recall_seen == 0:
        f1_score_seen = 0
    else:
        f1_score_seen = 2 * (precision_seen * recall_seen) / (precision_seen + recall_seen)
    # Calculate accuracy, precision, recall, f1-score of unseen words
    accuracy_unseen = (unseen_TP_count + unseen_TN_count) / (total_unseen)
    if unseen_TP_count + unseen_FP_count == 0:
        precision_unseen = 0
    else:
        precision_unseen = unseen_TP_count / (unseen_TP_count + unseen_FP_count)
    if unseen_TP_count + unseen_FN_count == 0:
        recall_unseen = 0
    else:
        recall_unseen = unseen_TP_count / (unseen_TP_count + unseen_FN_count)
    if precision_unseen + recall_unseen == 0:
        f1_score_unseen = 0
    else:
        f1_score_unseen = 2 * (precision_unseen * recall_unseen) / (precision_unseen + recall_unseen)
    # Calculate accuracy, precision, recall, f1-score of all words
    if total_seen + total_unseen == 0:
        accuracy_total = 0
    else:
        accuracy_total = (total_TP_count + total_TN_count) / (total_seen + total_unseen)
    if total_TP_count + total_FP_count == 0:
        precision_total = 0
    else:
        precision_total = total_TP_count / (total_TP_count + total_FP_count)
    if total_TP_count + total_FN_count == 0:
        recall_total = 0
    else:
        recall_total = total_TP_count / (total_TP_count + total_FN_count)
    if precision_total + recall_total == 0:
        f1_score_total = 0
    else:
        f1_score_total = 2 * (precision_total * recall_total) / (precision_total + recall_total)

    # Organize all results in a dictionary
    results = {'seen': {'accuracy': accuracy_seen, 'precision': precision_seen, 'recall': recall_seen,
                        'f1': f1_score_seen, 'cf_matrix': cf_matrix_seen},
               'unseen': {'accuracy': accuracy_unseen, 'precision': precision_unseen, 'recall': recall_unseen,
                          'f1': f1_score_unseen, 'cf_matrix': cf_matrix_unseen},
               'total': {'accuracy': accuracy_total, 'precision': precision_total, 'recall': recall_total,
                         'f1': f1_score_total, 'cf_matrix': cf_matrix_total}}
    # append all results to the file '{}_classification_report.txt'.format(experiment_name)
    with open(os.path.join(plots_dir, '{}_classification_report.txt'.format(experiment_name)), 'a') as f:
        f.write("\n\n\n")
        f.write("==============================================================\n")
        f.write("{} Manual Report:\n".format(experiment_name))
        f.write("\n")
        f.write("Seen: \n accuracy: {}\n precision: {}\n recall: {}\n f1: {}\n".format(accuracy_seen, precision_seen,
                                                                                          recall_seen, f1_score_seen))
        f.write("\n")
        f.write("Unseen: \n accuracy: {}\n precision: {}\n "
                "recall: {}\n f1: {}\n".format(accuracy_unseen, precision_unseen, recall_unseen, f1_score_unseen))
        f.write("\n")
        f.write("Total: \n accuracy: {}\n precision: {}\n "
                "recall: {}\n f1: {}\n".format(accuracy_total, precision_total, recall_total, f1_score_total))
        f.write("\n\n\n")

    # save all confusion matrices
    for result_type in results:
        cf_matrix = results[result_type]['cf_matrix']
        title = '{} {} Confusion Matrix'.format(experiment_name, result_type)
        filename = '{}_{}_confusion_matrix.png'.format(experiment_name, result_type)
        create_confusion_matrix(cf_matrix, title, filename, plots_dir)
    # save results as a pickle file
    with open(os.path.join(plots_dir, '{}_results.pkl'.format(experiment_name)), 'wb') as f:
        pickle.dump(results, f)

    print(results)


def evaluate_experiment(results_dir, experiment_name, corpus):
    if experiment_name in ['constant_false', 'majority']:
        file_type = 'json'
    else:
        file_type = 'pkl'
    predictions_path = '{}/{}/predictions.{}'.format(results_dir, experiment_name, file_type)
    plots_dir = '{}/{}/plots'.format(results_dir, experiment_name)
    evaluate_results(predictions_path, plots_dir, experiment_name)
    # load training data from json file
    train_dataset = pd.read_json('prepared_data/train/full_train_3_labels_{}.json'.format(corpus))
    words_statistics, metaphor_words, non_metaphor_words = \
        calculate_words_statistics(train_dataset)

    # Load predictions from pickle file
    if file_type == 'json':
        predictions = pd.read_json(predictions_path)
    else:
        predictions = pd.read_pickle(predictions_path)

    full_eval(predictions['data'], predictions['labels'], predictions['predictions'], metaphor_words,
                       plots_dir, experiment_name, non_metaphor_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classification model')
    parser.add_argument('--experiment', type=str, default='majority')
    parser.add_argument('--corpus', type=str, default='pre_piyut_1', choices=['pre_piyut_1',
                                                                               'pre_piyut_20',
                                                                               'pinchas_1',
                                                                               'all_1'])
    args = parser.parse_args()
    experiment_name = args.experiment
    corpus = args.corpus
    if experiment_name == 'majority' or experiment_name == 'constant_false':
        results_dir = 'results_vanilla_models'
    else:
        results_dir = 'results_michael'
    evaluate_experiment(results_dir, experiment_name, corpus)


