import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import matplotlib as mpl
import seaborn as sns
import regex as re
import argparse
from evaluation import calculate_words_statistics
sns.set()


def get_data(texts_file, annotations_file):
    """
    Load data from csv files
    :param texts_file: The file name
    :param annotations_file: Annotations file name
    :return: Annotations and texts dataframes
    """
    texts = pd.read_csv(texts_file)
    annotations = pd.read_csv(annotations_file)
    return annotations, texts


def check_word_len(fulltext, start_offset):
    """
    Returns the length of the word starting from start_offset in text
    :param fulltext: Full text
    :param start_offset: The offset of the word
    :return: Length of the word starting from start_offset
    """
    text = fulltext[start_offset:]
    words = text.split()
    return len(words[0])


def generate_data(data_sources, source_name, label_names, args, bad_data_list=None,
                  bad_text_id_list=None, ood_text_name_list=None):
    """
    Function that generates the data for the model
    param texts_file: path to the file with texts
    param annotations_file: path to the file with annotations
    param bad_data_list: list of texts that should be removed from the data
    param bad_text_id_list: list of text ids that should be removed from the data
    :return:
    Dataframe with columns: data: list of words, labels: list of labels
    """
    texts_file, annotations_file = data_sources[source_name]['path_to_texts'], \
                                   data_sources[source_name]['path_to_annotations']
    # Load training args from config.yaml file
    plots_dir = args.plots_dir
    if bad_data_list is None:
        bad_data_list = []
    if bad_text_id_list is None:
        bad_text_id_list = []
    if ood_text_name_list is None:
        ood_text_name_list = []

    # Get data from csv
    annotations, texts = get_data(texts_file, annotations_file)

    # iterate over rows in texts dataframe
    for index, row in texts.iterrows():
        fulltext = row.fulltext
        length = row.length
        fulltext_length = len(fulltext)
        if fulltext_length != length:
            fulltext_replaced = fulltext.replace('\n', '\r\n')
            new_length = len(fulltext_replaced)
            if new_length == length:
                texts.at[index, 'fulltext'] = fulltext_replaced
                # print('Replaced new line character in text name: {}'.format(row.name))
            else:
                if abs(new_length - length) < abs(fulltext_length - length):
                    texts.at[index, 'fulltext'] = fulltext_replaced
                else:
                    pass


    annotations.drop(annotations[annotations.id.isin(bad_data_list)].index, inplace=True)
    annotations.drop(annotations[annotations.text_id.isin(bad_text_id_list)].index,
                     inplace=True)

    # Get annotations
    tag_names = ['דימוי', 'הנפשה', 'כיוני פשט', 'כינוי', 'כינוי דימויי', 'כינוי מונפש', 'כינוי מטונימי',
                    'כינוי מטפורי', 'כינוי סינקדוכי', 'כינוי פשט', 'כינוי ציורי', 'מטונימיה', 'מטפורה',
                    'מטפוריקה', 'סינקדוכה', 'סמיכות', 'פועל', 'פועל + שם עצם', 'פרסוניפיקציה', 'שם עצם', 'שם תואר',
                    'תואר הפועל']

    tag_name_hist = {'דימוי': 0, 'הנפשה': 0, 'כיוני פשט': 0, 'כינוי': 0, 'כינוי דימויי': 0, 'כינוי מונפש': 0,
                    'כינוי מטונימי': 0, 'כינוי מטפורי': 0, 'כינוי סינקדוכי': 0, 'כינוי פשט': 0, 'כינוי ציורי': 0,
                    'מטונימיה': 0, 'מטפורה': 0, 'מטפוריקה': 0, 'סינקדוכה': 0, 'סמיכות': 0, 'פועל': 0,
                    'פועל + שם עצם': 0, 'פרסוניפיקציה': 0, 'שם עצם': 0, 'שם תואר': 0, 'תואר הפועל': 0}

    metaphorical_tags = ['כינוי ציורי', 'מטפוריקה']

    # fl_annotations = annotations.loc[(
    #             annotations.path.str.contains(metaphorical_tags[0], regex=False) |
    #             annotations.path.str.contains(metaphorical_tags[1], regex=False))]

    fl_annotations = annotations

    # Add columns for labels and text
    last_col_num = len(texts.columns) # Get the last column number
    texts.insert(last_col_num, "labels", "")  # Add column for labels
    texts.insert(last_col_num + 1, "data", "")  # Add column for data
    texts.insert(last_col_num + 2, "POS", "")  # Add column for POS
    texts.insert(last_col_num + 3, "metaphor_percentage", 0.0)
    texts.insert(last_col_num + 4, "text_size", 0)
    texts.insert(last_col_num + 5, "size_percentage_tuple", "")
    texts.labels = texts.labels.astype(object)  # Convert labels to object
    texts.data = texts.data.astype(object)  # Convert data to object
    texts.POS = texts.POS.astype(object)  # Convert POS to object
    texts.metaphor_percentage = texts.metaphor_percentage.astype(float)  # Convert metaphor_percentage to float
    texts.text_size = texts.text_size.astype(int)  # Convert text_size to int
    texts.size_percentage_tuple = texts.size_percentage_tuple.astype(object)

    for index, text in texts.iterrows():
        fulltext = texts.at[index, "fulltext"]  # Get fulltext
        if '\u202a' in fulltext or '\u202b' in fulltext or '\u202c' in fulltext:
            fulltext = fulltext.replace('\u202a', '\n')
            fulltext = fulltext.replace('\u202b', '\n')
            fulltext = fulltext.replace('\u202c', '\n')
        fulltext_split = fulltext.split()  # Split fulltext into words
        # delete empty strings
        fulltext_split = [x for x in fulltext_split if x != '']
        texts.at[index, "data"] = fulltext_split  # Add data to data column
        texts.at[index, "labels"] = np.zeros([len(fulltext_split)], dtype=int)  # Add labels to labels column
        texts.at[index, "POS"] = np.zeros([len(fulltext_split)])  # Add POS to POS column
        texts.at[index, "metaphor_percentage"] = 0.0
        texts.at[index, "text_size"] = len(fulltext_split)

    # For each annotation, add the tag
    for index, annotation in fl_annotations.iterrows():
        text_id = annotation['text_id']
        fulltext = texts[texts.id == text_id].fulltext.to_numpy()[0]

        start_offset = annotation['start_offset']
        end_offset = annotation['end_offset']
        metaphor_phrase = annotation.phrase

        for tag_name in tag_names:
            if annotation.path.find(tag_name) != -1:
                tag_name_hist[tag_name] += 1

        is_metaphor = False
        for metaphorical_tag in metaphorical_tags:
            if annotation.path.find(metaphorical_tag) != -1:
                is_metaphor = True
                break

        if not is_metaphor:
            continue

        if len(metaphor_phrase.strip()) <= 1:
            continue

        # Clean annotations from spaces at the end. update the end_offset and start_offset if needed
        space_list = ['\n', ' ', '\t', '\r', '\u202a', '\u202b', '\u202c', '"']

        # Check if start_offset is in the middle of a word
        while start_offset > 0 and fulltext[start_offset - 1] not in space_list:
            start_offset = start_offset - 1

        # Check if end_offset is in the middle of a word
        while end_offset < len(fulltext) and fulltext[end_offset] not in space_list:
            end_offset = end_offset + 1

        # check if the start_offset is a space
        while start_offset > 0 and fulltext[start_offset] in space_list:
            start_offset = start_offset + 1

        # check if the end_offset is a space
        while end_offset < len(fulltext) and fulltext[end_offset - 1] in space_list:
            end_offset = end_offset - 1
        initial_start_offset = start_offset
        initial_end_offset = end_offset
        initial_phrase = fulltext[start_offset:end_offset]
        counter_fixed_tags = 0
        if metaphor_phrase.strip() == fulltext[start_offset:end_offset] \
            or metaphor_phrase.strip() in fulltext[start_offset:end_offset] or \
                fulltext[start_offset:end_offset] in metaphor_phrase:
            pass
        else:  # fix the tag
            counter_fixed_tags += 1
            metaphor_phrase = metaphor_phrase.strip()
            # find the index of the metaphor phrase in the fulltext
            list_possible_start_idx = []
            search_start_index = 0
            word_index = fulltext.find(metaphor_phrase, search_start_index)
            while word_index != -1:
                list_possible_start_idx.append(word_index)
                search_start_index = word_index + 1
                word_index = fulltext.find(metaphor_phrase, search_start_index)
            start_offset = min(list_possible_start_idx, key=lambda x: abs(x - start_offset))
            end_offset = start_offset + len(fulltext[start_offset:].split()[0])
        # Check if start_offset is in the middle of a word
        while start_offset > 0 and fulltext[start_offset - 1] not in space_list:
            start_offset = start_offset - 1

        remaining_metaphor_len = end_offset - start_offset
        word_index = len(fulltext[:start_offset].replace('\u202b', '').replace('\u202c', '').split())
        number_of_words_in_phrase = 0
        next_word_start_index = start_offset

        # Iterate over the words in the phrase (annotation) and add the labels
        while remaining_metaphor_len > 0:
            if '\n' in fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len] or \
                    '\r' in fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len] or \
                        '\u202b' in fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len] or \
                            '\u202c' in fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len] or \
                                '\u202a' in fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len]:
                special_chars = fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len].count('\n')
                special_chars += fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len].count('\r')
                special_chars += fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len].count('\u202b')
                special_chars += fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len].count('\u202c')
                special_chars += fulltext[next_word_start_index:next_word_start_index+remaining_metaphor_len].count('\u202a')
                remaining_metaphor_len -= special_chars
            remaining_text = fulltext[next_word_start_index:]  # Get the remaining text
            word_len = len(remaining_text.split()[0])  # Get the length of the next word
            remaining_metaphor_len = remaining_metaphor_len - word_len - 1  # -1 for the space
            curr_word = fulltext[next_word_start_index:next_word_start_index + word_len]
            if '\n' in curr_word or '\r' in curr_word or ' ' in curr_word or '\u202b' in curr_word or '\u202c' in curr_word:
                special_chars = curr_word.count('\n')
                special_chars += curr_word.count('\r')
                special_chars += curr_word.count(' ')
                special_chars += curr_word.count('\u202b')
                special_chars += curr_word.count('\u202c')
                special_chars += curr_word.count('\u202a')
                curr_word = fulltext[next_word_start_index:next_word_start_index + word_len + special_chars]
                curr_word = curr_word.replace('\n', '')
                curr_word = curr_word.replace('\r', '')
                curr_word = curr_word.replace(' ', '')
                curr_word = curr_word.replace('\u202b', '')
                curr_word = curr_word.replace('\u202c', '')
                curr_word = curr_word.replace('\u202a', '')
            corresponding_data_word = texts.loc[texts.id == text_id, "data"].to_numpy()[0][word_index]
            if number_of_words_in_phrase == 0:  # If it's the first word in the phrase
                texts.loc[texts.id == text_id, "labels"].to_numpy()[0][word_index] = label_names["B-metaphor"]
                # check if th word is fulltext[start_offset:end_offset]
                if not corresponding_data_word == curr_word:
                    # raise Exception("The word in the text is not the same as the word in the annotation")
                    print("Data: {}, Corresponding label: {}".format(corresponding_data_word, curr_word))
                    print("The word in the text is not the same as the word in the annotation")
                # texts.loc[texts.id == text_id, "labels"].to_numpy()[0][word_index] = label_names["metaphor"]
            else:  # If it's not the first word in the phrase
                texts.loc[texts.id == text_id, "labels"].to_numpy()[0][word_index] = label_names["I-metaphor"]
                if not corresponding_data_word == curr_word:
                    # raise Exception
                    print("Data: {}, Corresponding label: {}".format(corresponding_data_word, curr_word))
                    print("The word in the text is not the same as the word in the annotation")
                # texts.loc[texts.id == text_id, "labels"].to_numpy()[0][word_index] = label_names["metaphor"]
                # TODO update POS
            # texts.loc[texts.id == text_id, "POS"].to_numpy()[0][word_index] = POS_names[]
            word_index = word_index + 1
            next_word_start_index = next_word_start_index + word_len + 1  # +1 for the space
            number_of_words_in_phrase = number_of_words_in_phrase + 1  # +1 for the word
    # clean texts from '\u202a', '\u202b', '\u202c'
    texts.data = texts.data.apply(lambda x: [word.replace('\u202a', '').replace('\u202b', '').replace('\u202c', '') for word in x])
    delete_list = ""
    for df_index, (text, labels, POS) in enumerate(zip(texts.data, texts.labels, texts.POS)):
        if len(text) != len(labels):
            raise Exception("Text and labels are not the same length")
        # delete words (with labels) that not contain letters א-ת
        for curr_index, word in enumerate(text):
            if not re.search('[א-ת]', word):
                delete_list = delete_list + word + ","
                text.pop(curr_index)
                # delete from numpy array
                labels = np.delete(labels, curr_index)
                POS = np.delete(POS, curr_index)
        texts.at[df_index, "data"] = text
        texts.at[df_index, "labels"] = labels
        texts.at[df_index, "POS"] = POS
    print("deleted words: " + delete_list)
    print("Done cleaning data")

    ood_texts = texts[texts['name'].isin(ood_text_name_list)]
    texts.drop(texts[texts['name'].isin(ood_text_name_list)].index, inplace=True)
    # plot tag histogram
    tag_names_reversed = [get_display(text) for text in tag_name_hist.keys()]
    plt.bar(tag_names_reversed, tag_name_hist.values())
    plt.xticks(rotation=90)
    plt.ylabel('# annotations')
    plt.xlabel('Tag')
    plt.title('Tag histogram in {}'.format(source_name))
    # add the values on top of the bars
    for i, v in enumerate(tag_name_hist.values()):
        plt.text(i - 0.25, v + 0.05, str(v))
    plt.tight_layout()
    plt.savefig('{}/tag_hist_{}.png'.format(plots_dir, source_name))
    plt.show()

    # show only  metaphorical_tags = ['כינוי ציורי', 'מטפוריקה']
    metaphorical_tags = ['כינוי ציורי', 'מטפוריקה']
    metaphorical_tags_hist = {tag_name: tag_name_hist[tag_name] for tag_name in metaphorical_tags}
    metaphorical_tags_hist_reversed = [get_display(text) for text in metaphorical_tags_hist.keys()]
    plt.bar(metaphorical_tags_hist_reversed, metaphorical_tags_hist.values())
    plt.xticks(rotation=90)
    plt.ylabel('# annotations')
    plt.xlabel('Tag')
    plt.title('Tag histogram in {}'.format(source_name))
    # add the values on top of the bars
    for i, v in enumerate(metaphorical_tags_hist.values()):
        plt.text(i - 0.25, v + 0.05, str(v))
    plt.tight_layout()
    plt.savefig('{}/metaphorical_tag_hist_{}.png'.format(plots_dir, source_name))
    plt.show()

    tag_histogram_dict = {}
    tag_histogram_array = []
    for text_name, labels in zip(texts['name'], texts['labels']):
        number_of_metaphors = (labels != 0).sum()
        metaphor_percentage = number_of_metaphors / len(labels)
        tag_histogram_dict[text_name] = metaphor_percentage
        tag_histogram_array.append(metaphor_percentage)
        texts.loc[texts.name == text_name, "metaphor_percentage"] = metaphor_percentage

    plt.hist(tag_histogram_array, bins=20)
    plt.ylabel('# texts')
    plt.xlabel('Tag ratio')
    plt.title('metaphor ratio in {}'.format(source_name))
    plt.tight_layout()
    plt.savefig('{}/metaphor_ratio_{}.png'.format(plots_dir, source_name))
    plt.show()

    text_size_labels = ["tiny", "small", "medium", "large", "huge"]
    if source_name == "pinchas":
        text_size_labels = ["small", "medium", "large"]
    text_size_q = len(text_size_labels)
    text_size_binned = pd.qcut(texts['text_size'], q=text_size_q, labels=text_size_labels)

    metaphor_percentage_labels = ["very low", "low", "medium", "high", "very high"]
    if source_name == "pinchas":
        metaphor_percentage_labels = ["low", "medium", "high"]
    metaphor_percentage_q = len(metaphor_percentage_labels)
    metaphor_percentage_binned = pd.qcut(texts['metaphor_percentage'], q=metaphor_percentage_q,
                                         labels=metaphor_percentage_labels,
                                         duplicates='drop')
    texts['size_percentage_tuple'] = list(zip(text_size_binned, metaphor_percentage_binned))

    tuple_size = texts.groupby(texts['size_percentage_tuple'], as_index=False).size()

    # plot texts['metaphor_percentage'] vs texts['text_size']
    plt.figure(figsize=(10, 5))
    plt.scatter(texts['text_size'], texts['metaphor_percentage'])
    plt.ylabel('metaphor ratio')
    plt.xlabel('text size (words)')
    plt.title('metaphor ratio vs text size in {}'.format(source_name))
    plt.tight_layout()
    plt.savefig('{}/metaphor_ratio_vs_text_size_{}.png'.format(plots_dir, source_name))
    plt.show()


    tag_histogram_array_ood = []
    for text_name, labels in zip(ood_texts['name'], ood_texts['labels']):
        counter = 0
        for label in labels:
            if label != 0:
                counter += 1
        tag_histogram_array_ood.append(counter / len(labels))

    plt.hist(tag_histogram_array_ood, bins=20)
    plt.ylabel('# texts')
    plt.xlabel('Tag ratio')
    plt.title('metaphor ratio in {} OOD'.format(source_name))
    plt.tight_layout()
    plt.savefig('{}/metaphor_ratio_ood.png'.format(plots_dir))
    plt.show()


    if len(ood_text_name_list) > 0:
        return texts, ood_texts
    else:
        return texts


def text_into_chunks(texts, rows_per_example=1, split_by="\r\n", per_word=False):
    """
    # Function that splits the data by rows
    :param texts: Dataframe with columns: data: list of words, labels: list of labels
    :return:
    texts_parts: Dataframe with columns:
                    data: list of words (corresponding to rows in the original text),
                    labels: list of labels
    """
    # initialize the dataframe
    # every sample is a full tagged sentence
    HP = pd.DataFrame(columns=['label', 'sentence', 'POS', 'numOfMetaphor'])
    if per_word:
        # every sample is a sentence with a word and it's label
        HP_per_word = pd.DataFrame(columns=['label', 'sentence', 'POS', 'w_index', 'word'])
    else:
        HP_per_word = None
    # texts_parts = pd.DataFrame(columns=['labels', 'data', 'id', 'name', 'user_id', 'corpus_id', 'genre_id', 'length'])

    max_number_of_tokens = 256
    for index, text in texts.iterrows():
        start_index = 0
        fulltext = text.fulltext.replace("\n", "\r\n")
        fulltext_split = fulltext.split(split_by)
        # clean fulltext_split from non-word examples
        fulltext_split = [sentence.strip() for sentence in fulltext_split]
        fulltext_split = [sentence for sentence in fulltext_split if not sentence.isspace() and sentence != '']
        part_index = 0
        while part_index < len(fulltext_split):
            curr_part = fulltext_split[part_index]
            curr_part_split = curr_part.split()
            curr_part_split = [word for word in curr_part_split if re.search('[א-ת]', word)]
            for number_of_rows in range(rows_per_example - 1):
                if part_index < len(fulltext_split) - 1:
                    next_part = fulltext_split[part_index + 1]
                    next_part_split = next_part.split()
                    next_part_split = [word for word in next_part_split if re.search('[א-ת]', word)]
                    curr_part_split = curr_part_split + next_part_split
                    part_index = part_index + 1
            number_of_words_in_part = len(curr_part_split)
            if number_of_words_in_part == 0:
                continue
            if number_of_words_in_part > max_number_of_tokens:
                raise Exception('The number of tokens in a sentence is bigger than the max number of tokens')
            # while number_of_words_in_part > max_number_of_tokens:
                # HPtok_next_free = HPtok.shape[0]
                # HPtok.loc[HPtok_next_free] = [text.labels[start_index:start_index + number_of_words_in_part],
                #                               text.data[start_index:start_index + number_of_words_in_part],
                #                               text.POS[start_index:start_index + number_of_words_in_part],
                #                               sum(text.labels[start_index:start_index + number_of_words_in_part])]
                # for word_index in range(number_of_words_in_part):
                #     HPall_next_free = HPall.shape[0]
                #     HPall.loc[HPall_next_free] = [text.labels[word_index],
                #                                   text.data[start_index:start_index + number_of_words_in_part],
                #                                   text.POS[word_index],
                #                                   word_index]
                # start_index = start_index + max_number_of_tokens
                # # Update the part to be the remaining part
                # part = " ".join(part.split()[max_number_of_tokens:])
                # number_of_words_in_part = len(part.split())
            HP_next_free = HP.shape[0]
            HP.loc[HP_next_free] = [text.labels[start_index:start_index + number_of_words_in_part],
                                          text.data[start_index:start_index + number_of_words_in_part],
                                          text.POS[start_index:start_index + number_of_words_in_part],
                                          sum(text.labels[start_index:start_index + number_of_words_in_part])]
            if per_word:
                for word_index in range(number_of_words_in_part):
                    HPall_next_free = HP_per_word.shape[0]
                    HP_per_word.loc[HPall_next_free] = [text.labels[word_index],
                                                    HP.loc[HP_next_free, 'sentence'],
                                                    HP.loc[HP_next_free, 'POS'][word_index],
                                                    word_index,
                                                    HP.loc[HP_next_free, 'sentence'][word_index]
                                                  ]
            part_index = part_index + rows_per_example
            start_index = start_index + number_of_words_in_part
    return HP, HP_per_word


def make_binary_labels(df):
    for index, row in df.iterrows():
        labels = row.labels
        for i in range(len(labels)):
            if labels[i] == 2:
                labels[i] = 1
    return df


def prepare_data(args):
    path_to_prepared_data = args.path_to_prepared_data
    train_validation_split = args.train_validation_split
    binary_labels = args.binary_l
    test_size = args.test_size
    random_state = args.random_state
    rows_per_example = args.rows_per_example
    generate_per_word = args.generate_per_word
    plots_dir = args.plots_dir
    corpus = args.corpus

    # if folders not exist, create them
    if not os.path.exists(path_to_prepared_data):
        os.makedirs(path_to_prepared_data)
    if not os.path.exists(path_to_prepared_data + '/train'):
        os.makedirs(path_to_prepared_data + '/train')
    if not os.path.exists(path_to_prepared_data + '/test'):
        os.makedirs(path_to_prepared_data + '/test')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    data_sources = {'piyyut': {'path_to_texts': "raw_data/piyyut_texts.csv",
                               'path_to_annotations': "raw_data/piyyut_annotations.csv"},
                    'pinchas': {'path_to_texts': "raw_data/pinechas_hacohen_texts.csv",
                                        'path_to_annotations': "raw_data/pinechas_hacohen_annotations.csv"},
                    'yose_ben_yose': {'path_to_texts': "raw_data/yose_ben_yose_texts.csv",
                                     'path_to_annotations': "raw_data/yose_ben_yose_annotations.csv"}
                    }

    # Load and create a dataframe with the data and the labels
    piyyut_bad_data_list = [52341, 52586]
    piyyut_bad_text_id_list = ['VC26-CATMA_65A3AE5A-6076-4F62-ADA8-81B98EC4286F',
                        'VC26-CATMA_AA7C8A80-CFA9-4DB7-8924-E04BB61E683C']
    piyyut_ood_text_name_list = ['אשר קדש ידיד', 'אשר צג אגוז', 'ישמח משה במתנת חלקו', 'לך נודה תמיד', 'אם תעינו לא תתעינו',
                          'לא ארמון על משפטו', 'לא אורים ותומים על לב כהן', 'לא אישים ולא אשם', 'אתה הנחלת תורה לעמך',
                          'אל אדון', 'תקנת שבת', 'אל אשר שבת', 'אמת אמונתך בשביעי קימת', 'אתה הכנת מנוחת שבת',
                          'אתה מנהיג', 'יום ענוגה תתה', 'תשועה שלמה', 'אהובים בחר גאולים דרש', 'אות במועד גש',
                          'אז כל מפעל', 'אז שרו עם', 'תחילת כל מעש', 'אל ברוך גדול דעה', 'שנת אוצרך הטוב',
                          'אל ברוך גדול דעה', 'אותות בוננת גילת דודים', 'בורא עולם']
    if binary_labels:
        label_names = {"O": 0, "B-metaphor": 1, "I-metaphor": 1}
    else:
        label_names = {"O": 0, "B-metaphor": 1, "I-metaphor": 2}
    if corpus == 'pre_piyut' or corpus == 'all':
        piyyut_texts, piyyut_ood_texts = generate_data(data_sources,
                                                   'piyyut',
                                                    label_names,
                                                    args,
                                                    piyyut_bad_data_list,
                                                    piyyut_bad_text_id_list,
                                                    piyyut_ood_text_name_list)

        print("Piyyut texts len:", len(piyyut_texts))
        # Fix one of the data examples in 'Piyyut'
        piyyut_texts[piyyut_texts['id'] == 'VC26-CATMA_026AC456-6894-4E6D-9B3E-57B7EF58B3C2']['data'][85][17] = 'ואל'

        yose_ben_yose_texts = generate_data(data_sources,
                                        'yose_ben_yose',
                                        label_names,
                                        args)
        print("Yose ben Yose texts len:", len(yose_ben_yose_texts))

    # Merge the dataframes
    if corpus == 'all':
        pinchas = generate_data(data_sources, 'pinchas', label_names, args)
        texts = pd.concat([piyyut_texts, yose_ben_yose_texts, pinchas], ignore_index=True)
        print("All texts len:", len(texts))
    elif corpus == 'pinchas':
        texts = generate_data(data_sources, 'pinchas', label_names, args)
        print("Pinchas texts len:", len(texts))
    else:  # corpus == 'pre_piyut'
        texts = pd.concat([piyyut_texts, yose_ben_yose_texts], ignore_index=True)
        print("Pre piyyut texts len:", len(texts))

    # remove duplicates
    texts = texts.drop_duplicates(subset=['name'])
    texts = texts[texts['name'].str.contains('אבן מעמסה - נוסח ב') == False]
    texts = texts[texts['name'].str.contains('אבן מעמסה - נוסח ג') == False]
    texts = texts.reset_index(drop=True)

    full_train, test = train_test_split(texts, test_size=test_size, random_state=random_state,
                                        stratify=texts['size_percentage_tuple'])


    print("Train len:", len(full_train))
    print("Test len:", len(test))

    with open('{}/{}_test_train_len.txt'.format(plots_dir,corpus), 'w') as f:
        f.write("Train len: {} Test len: {}".format(len(full_train), len(test)))

    tag_histogram_array_test = []
    for text_name, labels in zip(test['name'], test['labels']):
        counter = 0
        for label in labels:
            if label != 0:
                counter += 1
        tag_histogram_array_test.append(counter / len(labels))
        if counter / len(labels) > 0.6 or counter / len(labels) < 0.01:
            print("text name: {}, tag ratio: {}".format(text_name, counter / len(labels)))
    plt.clf()
    plt.hist(tag_histogram_array_test, bins=20)
    plt.ylabel('# texts')
    plt.xlabel('Tag ratio')
    plt.title('metaphor ratio in test - {}'.format(corpus))
    plt.tight_layout()
    plt.savefig('metaphor_ratio_test_{}.png'.format(corpus))
    plt.show()

    HP_all, HP_per_word_all = text_into_chunks(texts, rows_per_example, per_word=generate_per_word)

    HP_test, HP_per_word_test = text_into_chunks(test, rows_per_example, per_word=generate_per_word)
    HP_full_train, HP_per_word_full_train = text_into_chunks(full_train, rows_per_example, per_word=generate_per_word)

    train_words_statistics, train_metaphor_words, train_non_metaphor_words = calculate_words_statistics(HP_full_train)
    test_words_statistics, test_metaphor_words, test_non_metaphor_words = calculate_words_statistics(HP_test)
    # Count the number of unique words in the train and test sets
    train_metaphor_words_set = set(train_metaphor_words)
    train_unique_words_num = len(train_metaphor_words_set)

    test_metaphor_words_set = set(test_metaphor_words)
    test_unique_words_num = len(test_metaphor_words_set)
    print("Train unique words num:", train_unique_words_num)
    print("Test unique words num:", test_unique_words_num)

    # Count the number of metaphor words in the test set that are not in the train set
    new_metaphors = sum([1 for word in test_metaphor_words if word not in train_metaphor_words])
    print("New metaphor words in test set:", new_metaphors)

    # Count the number of metaphor words in the test set that are in the train set
    old_metaphors = sum([1 for word in test_metaphor_words if word in train_metaphor_words])
    print("Repeated metaphor words in test set:", old_metaphors)

    print("Test len:", len(HP_test))
    print("Full Train len:", len(HP_full_train))
    # generate a txt file with the Test and Train data lengths in the plots folder
    with open('{}/{}_test_train_len.txt'.format(plots_dir,corpus), 'w') as f:
        f.write("Test len: {}\n".format(len(HP_test)))
        f.write("Train len: {}".format(len(HP_full_train)))


    ending_name = '_{}_{}'.format(corpus, rows_per_example)

    if binary_labels:
        HP_test.to_json("{}/test/test{}.json".format(
            path_to_prepared_data, ending_name), orient='records')
        HP_full_train.to_json("{}/train/full_train{}.json".format(path_to_prepared_data, ending_name), orient='records')
        if generate_per_word:
            HP_per_word_test.to_json("{}/test/per_word_test{}.json".format(
                path_to_prepared_data, ending_name), orient='records')
            HP_per_word_full_train.to_json("{}/train/per_word_full_train{}.json".format(
                path_to_prepared_data, ending_name), orient='records')
    else:
        HP_test.to_json("{}/test/test_3_labels{}.json".format(
            path_to_prepared_data, ending_name), orient='records')
        HP_full_train.to_json("{}/train/full_train_3_labels{}.json".format(
            path_to_prepared_data, ending_name), orient='records')
        if generate_per_word:
            HP_per_word_test.to_json("{}/test/per_word_test_3_labels{}.json".format(
                path_to_prepared_data, ending_name), orient='records')
            HP_per_word_full_train.to_json("{}/train/per_word_full_train_3_labels{}.json".format(
                path_to_prepared_data, ending_name), orient='records')

    if train_validation_split > 0:
        # Split the data into train and validation and test
        # val_random_state = 1
        train, validation = train_test_split(full_train, test_size=train_validation_split,
                                             random_state=random_state,
                                             stratify=full_train['size_percentage_tuple'])
        print("Train len:", len(train))
        print("Validation len:", len(validation))

        tag_histogram_array_train = []
        for text_name, labels in zip(train['name'], train['labels']):
            counter = 0
            for label in labels:
                if label != 0:
                    counter += 1
            tag_histogram_array_train.append(counter / len(labels))
            if counter / len(labels) > 0.8 or counter / len(labels) < 0.01:
                print("text name: {}, tag ratio: {}".format(text_name, counter / len(labels)))
        plt.clf()
        plt.hist(tag_histogram_array_train, bins=20)
        plt.ylabel('# texts')
        plt.xlabel('Tag ratio')
        plt.title('metaphor ratio in train - {}'.format(corpus))
        plt.tight_layout()
        plt.savefig('metaphor_ratio_train_{}.png'.format(corpus))
        plt.show()

        tag_histogram_array_validation = []
        for text_name, labels in zip(validation['name'], validation['labels']):
            counter = 0
            for label in labels:
                if label != 0:
                    counter += 1
            tag_histogram_array_validation.append(counter / len(labels))
            if counter / len(labels) > 0.8 or counter / len(labels) < 0.01:
                print("text name: {}, tag ratio: {}".format(text_name, counter / len(labels)))
        plt.clf()
        plt.hist(tag_histogram_array_validation, bins=20)
        plt.ylabel('# texts')
        plt.xlabel('Tag ratio')
        plt.title('metaphor ratio in validation - {}'.format(corpus))
        plt.tight_layout()
        plt.savefig('metaphor_ratio_validation_{}.png'.format(corpus))
        plt.show()


        # print train.name.values into a txt file 'train_names.txt'
        with open("{}/train/train_names_{}.txt".format(path_to_prepared_data, random_state), "w") as f:
            for name in train.name.values:
                f.write(name + "\n")
        # print validation.name.values into a txt file 'validation_names.txt'
        with open("{}/train/validation_names_{}.txt".format(path_to_prepared_data, random_state), "w") as f:
            for name in validation.name.values:
                f.write(name + "\n")
        # print test.name.values into a txt file 'test_names.txt'
        with open("{}/test/test_names_{}.txt".format(path_to_prepared_data, random_state), "w") as f:
            for name in test.name.values:
                f.write(name + "\n")

        # Plot statistics about the data
        # count the number of words in each set
        train_words_num = train.data.apply(lambda x: len(x)).sum()
        validation_words_num = validation.data.apply(lambda x: len(x)).sum()
        test_words_num = test.data.apply(lambda x: len(x)).sum()
        if corpus == 'pre_piyut' or corpus == 'all':
            ood_words_num = piyyut_ood_texts.data.apply(lambda x: len(x)).sum()

        word_counts = [train_words_num, validation_words_num, test_words_num]

        # count the number of examples that is 1 or 2
        train_metaphor_num = train.labels.apply(lambda x: len([i for i in x if i == 1 or i == 2])).sum()
        validation_metaphor_num = validation.labels.apply(lambda x: len([i for i in x if i == 1 or i == 2])).sum()
        test_metaphor_num = test.labels.apply(lambda x: len([i for i in x if i == 1 or i == 2])).sum()
        if corpus == 'pre_piyut' or corpus == 'all':
            ood_metaphor_num = piyyut_ood_texts.labels.apply(lambda x: len([i for i in x if i == 1 or i == 2])).sum()
        if corpus == 'pre_piyut' or corpus == 'all':
            metaphor_counts = [train_metaphor_num, validation_metaphor_num, test_metaphor_num, ood_metaphor_num]
        else:
            metaphor_counts = [train_metaphor_num, validation_metaphor_num, test_metaphor_num]

        print("Train metaphor num:", train_metaphor_num)
        print("Validation metaphor num:", validation_metaphor_num)
        print("Test metaphor num:", test_metaphor_num)

        print("Train words num:", train_words_num)
        print("Validation words num:", validation_words_num)
        print("Test words num:", test_words_num)

        # bar plot: stacked barplot of the number of words and metaphors in each set
        fig, ax = plt.subplots(figsize=(10, 5))
        # ax.bar(["train", "validation", "test", "ood"], [train_words_num, validation_words_num, test_words_num, ood_words_num], color='c')
        ax.bar(["train", "validation", "test"], [train_words_num, validation_words_num, test_words_num], color='c')
        # ax.bar(["train", "validation", "test", "ood"], [train_metaphor_num, validation_metaphor_num, test_metaphor_num, ood_metaphor_num], color='m')
        ax.bar(["train", "validation", "test"], [train_metaphor_num, validation_metaphor_num, test_metaphor_num], color='m')
        ax.legend(["words", "metaphors"])
        ax.set_title("Number of words and metaphors in each split - {}".format(corpus))
        ax.set_xlabel("set")
        ax.set_ylabel("number of words and metaphors")
        # show the number of words in each set on top of the bars
        for i, v in enumerate(word_counts):
            ax.text(i, v, str(v), color='black', fontweight='bold')
        for i, (words_num, metaphors_num) in enumerate(zip(word_counts, metaphor_counts)):
            ax.text(i, metaphors_num, str(round(metaphors_num / words_num, 2)), color='black', fontweight='bold')
        plt.tight_layout()
        plt.savefig("{}/words_num_{}_{}.png".format(plots_dir, random_state, corpus))
        plt.show()

        HP_train, HP_per_word_train = text_into_chunks(train, rows_per_example)
        HP_validation, HP_per_word_validation = text_into_chunks(validation, rows_per_example)

        # Save the dataframe
        if binary_labels:
            HP_train.to_json("{}/train/train{}.json".format(path_to_prepared_data, ending_name), orient='records')
            HP_validation.to_json("{}/train/validation{}.json".format(path_to_prepared_data, ending_name), orient='records')
            if generate_per_word:
                HP_per_word_train.to_json("{}/train/per_word_train{}.json".format(path_to_prepared_data, ending_name), orient='records')
                HP_per_word_validation.to_json("{}/train/per_word_validation{}.json".format(path_to_prepared_data, ending_name), orient='records')
        else:
            HP_train.to_json("{}/train/train_3_labels{}.json".format(path_to_prepared_data, ending_name), orient='records')
            HP_validation.to_json("{}/train/validation_3_labels{}.json".format(path_to_prepared_data, ending_name), orient='records')
            if generate_per_word:
                HP_per_word_train.to_json("{}/train/per_word_train_3_labels{}.json".format(path_to_prepared_data, ending_name), orient='records')
                HP_per_word_validation.to_json("{}/train/per_word_validation_3_labels{}.json".format(path_to_prepared_data, ending_name), orient='records')
    # print the number of examples in each set
    print("train: {} \ntest: {}".format(len(HP_full_train), len(HP_test)))
    if train_validation_split > 0:

        print("validation: {}".format(len(HP_validation)))
    print_examples = False
    if print_examples:
        import highlight_text
        from highlight_text import HighlightText, ax_text, fig_text
        fig, ax = plt.subplots()
        # remove all the axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        first_idx = 0
        last_idx = 10

        examples_idx = range(first_idx, last_idx)
        examples_string = ""
        metaphors_num = 0
        for example_idx in examples_idx:
            sentence = HP_full_train['sentence'].values[example_idx]
            labels = HP_full_train['label'].values[example_idx]
            print(sentence, labels)
            example = ""
            for word, label in zip(sentence[::-1], labels[::-1]):
                # reverse word
                word = word[::-1]
                if label != 0:
                    example += f"<{word}> "
                    metaphors_num += 1
                else:
                    example += f"{word} "
            examples_string += f"{'':>12} {example}\n"

        highlight_textprops = []
        for i in range(metaphors_num):
            highlight_textprops.append({"color": "red"})

        # You can either create a HighlightTe]xt object
        HighlightText(x=0.0, y=1.0,
                      s=examples_string,
                      highlight_textprops=highlight_textprops,
                      ax=ax)
        # show
        plt.show()
    print("Done")


if __name__ == "__main__":

    plots_dir = 'plots'

    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--path_to_data', type=str, default='raw_data', help='Path to the data folder')
    parser.add_argument('--path_to_prepared_data', type=str, default='prepared_data',
                        help='Path to the prepared data folder')
    parser.add_argument('--binary_l', type=bool, default=False, help='Use binary labels')
    parser.add_argument('--random_state', type=int, default=1, help='Random state')
    parser.add_argument('--rows_per_example', type=int, default=1, help='Rows per example')
    parser.add_argument('--plots_dir', type=str, default=plots_dir, help='Path to the plots folder')
    parser.add_argument('--train_validation_split', type=float, default=0.2, help='Train validation split')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--generate_per_word', type=bool, default=False, help='Per word')
    parser.add_argument('--corpus', type=str, default='pre_piyut', help='Corpus',
                        choices=['pre_piyut', 'pinchas', 'all'])

    args = parser.parse_args()
    prepare_data(args)
