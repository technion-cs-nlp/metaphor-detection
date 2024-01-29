import math
import os
import sys
import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt
from bidi.algorithm import get_display
import requests
from git import Repo
import git

def prepare_data_BY(output):
    """
    This function prepares the data for the model.
    """
    generate_data_from_git = True
    generate_data_from_fl_dataset = True


    fl_data_paths = {'piyyut':
                         'raw_data/piyyut_texts.csv',
                     'pinechas':
                         'raw_data/pinechas_hacohen_texts.csv',
                     'yose':
                         'raw_data/yose_ben_yose_texts.csv'
                     }

    if generate_data_from_git:
        if not os.path.exists('public_domain_dump'):
            print("Downloading from git...")
            Repo.clone_from("https://github.com/projectbenyehuda/public_domain_dump.git",
                        "public_domain_dump")
            print("Downloaded from git.")
        path_to_data = 'public_domain_dump/txt_stripped'
        path_to_pseudocatalogue = 'public_domain_dump/pseudocatalogue.csv'

        # load pseudocatalogue from csv
        catalog_df = pd.read_csv(path_to_pseudocatalogue)
        # get all unique pseudocatalogue entries in genre
        genres = catalog_df['genre'].unique()
        df_data = pd.DataFrame(columns=['text', 'genre'])
        # iterate over all files in catalog_df
        for index, row in catalog_df.iterrows():
            # get the path to the file
            path_to_file = os.path.join(path_to_data, row['path'][1:] + '.txt')
            # read the file
            with open(path_to_file, 'r') as f:
                text = f.read()
            # clean the text
            text = text.split('את הטקסט לעיל הפיקו מתנדבי פרויקט בן־יהודה באינטרנט.  הוא זמין תמיד בכתובת הבאה:')[0]
            text = text.strip()
            # get the genre of the file
            genre = row['genre']
            # add the text and genre to the dataframe
            df_data = pd.concat([df_data, pd.DataFrame([[text, genre]], columns=['text', 'genre'])],
                                ignore_index=True)
        # split the dataframe into seperate dataframes for each
        dict_genres_to_en = {'שירה': 'poetry',
                             'זכרונות ויומנים': 'diaries',
                             'פרוזה': 'prose',
                             'משלים': 'parables',
                             'מאמרים ומסות': 'articles&essays',
                             'מילונים ולקסיקונים': 'dictionaries&lexicons',
                             'מחזות': 'plays',
                             'עיון': 'search',
                             'מכתבים': 'letters'}

        dict_genres_to_heb = {}
        for key, value in dict_genres_to_en.items():
            dict_genres_to_heb[value] = key
            dict_genres_to_heb[key] = value

        words_counter_dict = {}
        for genre in genres:
            df_genre = df_data[df_data['genre'] == genre]
            # remove 'genre' column from csv
            df_genre = df_genre.drop(columns=['genre'])
            # count the words in the text
            words_counter_dict[genre] = df_genre['text'].str.split().apply(len).sum()
            # save the dataframe to csv
            df_genre.to_csv('{}/{}.csv'.format(output, dict_genres_to_en[genre]),
                            index=False)
        # save full dataframe to csv
        df_data = df_data.drop(columns=['genre'])
        # df_data.to_csv('{}/unordered.csv'.format(output), index=False)

        # Generate data from the labeled data for the unsupervised learning
        # load csv
        supervised_words_counter_dict = {}
        for name, path in fl_data_paths.items():
            df_data = pd.read_csv(path)
            # keep only the 'fulltext' column
            df_data = df_data[['fulltext']]
            # change the column name to 'text'
            df_data.columns = ['text']
            # save the dataframe to csv
            # df_data.to_csv('/home/tok/figurative-language/data/fl_unsupervised.csv', index=False)
            supervised_words_counter_dict[name] = df_data['text'].str.split().apply(len).sum()

        words_counter_dict["supervised"] = sum(supervised_words_counter_dict.values())

        names = list(words_counter_dict.keys())
        values = list(words_counter_dict.values())
        # reverse text (end to start because this is hebrew)
        # names = [get_display(name) for name in names]
        # count the sum of the words in all genres
        num_words = sum(values)

        # convert only hebrew names to english, keep the rest as is
        names = [dict_genres_to_en[name] if name in dict_genres_to_heb.keys() else name for name in names]

        # plot the data in orange
        plt.bar(names, values, color='orange')
        plt.title('Number of words in each corpora')
        plt.xlabel('Genre')
        # Number of words in thousands
        plt.ylabel('Number of words (in thousands)')
        # add the total sum of the words to top right
        plt.text(len(words_counter_dict) - 0.5, max(values) + 20, 'Total: {:}'.format(int(num_words / 1000)),
                 horizontalalignment='right', verticalalignment='top', fontsize=12)
        # add the number of the words for each genre above the bar
        for i, v in enumerate(values):
            plt.text(i, v + 10, '{:}'.format(int(v / 1000)), horizontalalignment='center', verticalalignment='top', fontsize=8)
        # make xticks vertical
        plt.xticks(rotation=90)
        plt.tight_layout()
        # save the plot to DataExploration/plots folder. Create the folder if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/words_in_genres.png')
        # plt.show()

    if generate_data_from_fl_dataset:
        # Generate data from the labeled data for the unsupervised learning
        # load csv
        for name, path in fl_data_paths.items():
            df_data = pd.read_csv(path)
            # keep only the 'fulltext' column
            df_data = df_data[['fulltext']]
            # change the column name to 'text'
            df_data.columns = ['text']
            # save the dataframe to csv
            df_data.to_csv('{}/{}.csv'.format(output, name), index=False)


def prepare_data_Midrash(output):
    words_counter_dict = {}
    # iterate over the csvs in texts-language-model folder
    for file in os.listdir('texts-language-model'):
        # load the csv
        df_data = pd.read_csv('texts-language-model/' + file)
        # keep only the 'text' column
        df_data = df_data[['text']]
        # count the words in the text
        words_counter_dict[file.split('.')[0]] = df_data['text'].str.split().apply(len).sum()
        df_data.to_csv('{}/{}'.format(output, file), index=False)

    names = list(words_counter_dict.keys())
    values = list(words_counter_dict.values())
    # reverse text (end to start because this is hebrew)
    names = [get_display(name) for name in names]
    # count the sum of the words in all genres
    num_words = sum(values)

    # plot the data in green
    plt.bar(names, values, color='green')
    plt.title('Number of words in each corpora [thousands]')
    plt.xlabel('Genre')
    plt.ylabel('Number of words [thousands]')
    # add the total sum of the words to top left
    plt.text(len(words_counter_dict) - 0.5, max(values) + 10, 'Total: {:}'.format((num_words / 1000)),
             horizontalalignment='left', verticalalignment='top', fontsize=12)
    # add the number of the words for each genre above the bar
    for i, v in enumerate(values):
        plt.text(i, v + 10, '{:}'.format(v / 1000), horizontalalignment='center', verticalalignment='top', fontsize=8)
    # make xticks vertical
    plt.xticks(rotation=90)
    plt.tight_layout()
    # save the plot to DataExploration/plots folder. Create the folder if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/words_in_genres.png')
    # plt.show()


if __name__ == '__main__':
    print('Preparing data...')
    output_data_folder = 'mlm_data'
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)
    prepare_data_BY(output=output_data_folder)
    # prepare_data_Midrash(output=output_data_folder)
    print('Done!')
