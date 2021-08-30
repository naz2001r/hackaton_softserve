import os
import re
import json
import numpy as np
import time
from subprocess import call
from spacy.lang.en import English
from gensim.models import Word2Vec
from dataset import create_dataframe, split_sentence, split_words
from modelling import preparing_data_w2v
#from prediction import search_similar_text
from processing import get_text, get_files_path
from vectors import get_avg_vector, postprocess_vectors
import boto3
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from functools import partial
from tqdm import tqdm
from nltk.util import ngrams
from collections import Counter
from itertools import chain
from gensim.models.fasttext import FastText



def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

def create_dataset(data_root, folder_list, number_processes,path_json_processed, path_dataset_texts, nlp_eng ):
    # new folder with processed json

    files_list_dict = get_files_path(data_root, folder_list)
    partial_args = partial(get_text, folder_save=path_json_processed, lang_threshold=0.5)
    with Pool(processes=number_processes) as pool:
        pool.map(partial_args, list(files_list_dict.values()))

    # create dataset
    dataframe_list = []
    folder_exist_list = [item.split('.')[0] for item in os.listdir(path_json_processed )]

    for folder in tqdm(folder_exist_list):
        if folder in folder_list:
            file_processed_path = '{}/{}.json'.format(path_json_processed, folder)
            dataframe_list.append(
                create_dataframe(file_processed_path, doc_path_folder='s3://hackathon-baseline/duai_docs/'))
    data = pd.concat(dataframe_list).reset_index(drop=True)
    print('There are {} rows in created DataFrame'.format(data.shape[0]))

    #for test in local
    #data = data[:100]

    # split text into sentences
    data['text_sentences'] = apply_by_multiprocessing(data['text'],
                                                      partial(split_sentence, nlp=nlp_eng, sentence_length=10),
                                                      workers=number_processes)

    # split sentences into words
    data['text_words'] = data['text_sentences'].apply(lambda t: [split_words(sent, nlp_eng, 2) for sent in t])

    # remove records without sentences
    data = data[data['text_words'].apply(len) != 0].reset_index(drop=True)


    data.to_pickle(path_dataset_texts)

def create_data_name(data_root,path_dataset_names):
    WORD_BLACK_LIST = ['', 'installation', 'manual', 'manuals', 'en', 'transmitters', 'ultrasonic',
                       'quick', 'start', 'foundation', 'model', 'guide', 'rev', 'series', 'gas', 'sensor',
                       'transmitter', 'meters', 'pressure', 'supplement', 'shafer', 'replacement', 'protocol',
                       'instructions', 'instructions', 'service', 'control', 'configuration', 'operation',
                       'power', 'procedure', 'instruction', 'maintenance', 'level', 'guides', 'meter', 'analyzer']

    file_dict = {folder: os.listdir(os.path.join(data_root, folder)) for folder in os.listdir(data_root)} #[1:] for local

    def filter_word_freq(freq_dict, threshold):
        return {item: freq for item, freq in freq_dict.items() if freq > threshold}

    def sort_dict(freq_dict):
        return {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}

    device_model_pattern = r'\d{3,4}\w{0,2}'
    words = '-'.join(list(chain(*file_dict.values())))

    device_models_freq = Counter(ngrams(re.findall(device_model_pattern, words), 1))
    device_models_freq = sort_dict(filter_word_freq(device_models_freq, 1))

    unigram_freq = Counter(ngrams([w for w in re.sub(device_model_pattern, '', words).replace('.pdf', '').split('-') if
                                   w not in WORD_BLACK_LIST and len(w) > 4], 1))
    unigram_freq = filter_word_freq(unigram_freq, 8)
    bigram_freq = Counter(ngrams([w for w in re.sub(device_model_pattern, '', words).replace('.pdf', '').split('-') if
                                  w not in WORD_BLACK_LIST and len(w) > 4], 2))
    bigram_freq = filter_word_freq(bigram_freq, 4)

    unigram2del = []
    for manufacturer, freq in bigram_freq.items():
        w1 = (manufacturer[0],)
        w2 = (manufacturer[1],)
        if w1 in unigram_freq.keys() and w2 in unigram_freq.keys():
            if freq >= unigram_freq[w1] or freq >= unigram_freq[w2]:
                unigram2del.extend([w1, w2])
    clean_unigram_freq = {manufacturer: freq for manufacturer, freq in
                          unigram_freq.items() if manufacturer not in unigram2del}

    manufacturer_freq = sort_dict(dict(chain.from_iterable(d.items() for d in (unigram_freq, bigram_freq))))

    doc_map = []
    for folder, docs in file_dict.items():
        for doc in docs:
            clean_doc = doc.replace('-', ' ')

            device_model = []
            prev_model_freq = 0
            for prob_device_model, freq in device_models_freq.items():
                if (prob_device_model[0] in clean_doc) and (freq >= prev_model_freq):
                    device_model.append(prob_device_model[0])
                    prev_model_freq = freq
                elif freq < prev_model_freq:
                    break

            manufacturer = ''
            for prob_manufacturer in manufacturer_freq.keys():
                prob_manufacturer = ' '.join(prob_manufacturer)
                if prob_manufacturer in clean_doc:
                    manufacturer = prob_manufacturer
                    break

            doc_map.append(
                {'doc_name': doc, 'doc_class': folder, 'manufacturer': manufacturer, 'device_model': device_model})

            data_names = pd.DataFrame(doc_map)
            data_names.to_pickle(path_dataset_names)

def train_model(
    model_dir="/home/model-server", data_root="/home/model-server/text_json"
):
    """
    Method that trains model and stores artifacts under @model_dir
    :param model_dir: path to root model dir
    """
    # TODO: Replace the following with your implementation  ###
    # mock an artifact of default predictions

    #train in local
    #model_dir = './'
    #data_root = 'baseline/text_json'


    #number of free cores
    cores = multiprocessing.cpu_count()
    number_processes = cores - 2

    # spacy rule-based matching
    nlp_eng = English()
    sentencizer = nlp_eng.create_pipe("sentencizer")
    nlp_eng.add_pipe(sentencizer)


    folder_list = [
        'Ashcroft',
        'Density&Viscosity',
        'Flow',
        'Gas_analysis',
        'Level',
        'Liquid _analysis',
        'Pressure',
        'Temperature',
        'Valves_actuators'
    ]

    folder_data = model_dir
    folder_created_data = model_dir

    folder_json_processed = 'data_text_processed'
    path_documents = './documents'

    dataset_texts = 'data_texts.pkl'
    dataset_names = 'device2document_map.pkl'
    dataset_vectors = 'data_vectors.pkl'
    dataset_vectors_2 = 'data_vectors_2.pkl'
    model_name_w2v = 'word2vec.model'
    model_name_ft = 'fasttext.model'

    path_json_processed = os.path.join(folder_created_data, folder_json_processed)

    path_dataset_texts = os.path.join(folder_created_data, dataset_texts)
    path_dataset_names = os.path.join(folder_data, dataset_names)
    path_dataset_vectors = os.path.join(folder_created_data, dataset_vectors)
    path_dataset_vectors_2 = os.path.join(folder_created_data, dataset_vectors_2)
    path_model_w2v = os.path.join(folder_created_data, model_name_w2v)
    path_model_ft = os.path.join(folder_created_data, model_name_ft)

    #create dataset and save in path_dataset_texts
    create_dataset(data_root, folder_list, number_processes, path_json_processed, path_dataset_texts, nlp_eng)
    #create data names
    create_data_name(data_root, path_dataset_names)

    # dataset initialization
    data = pd.read_pickle(path_dataset_texts)


    sentences_list = preparing_data_w2v(data)
    sentences_list = [[word for word in sent if len(word)  > 2 ]
                      for sent in sentences_list]

    model_w2v = Word2Vec(min_count=4,
                         window=3,
                         size=300,
                         sample=6e-5,
                         alpha=0.025,
                         min_alpha=0.0001,
                         negative=20,
                         workers=cores-2)

    model_ft = FastText(min_count=3,
                        window=4,
                        size=300,
                        sample=6e-5,
                        alpha=0.025,
                        min_alpha=0.0001,
                        negative=20,
                        workers=cores-2
                        )

    model_w2v.build_vocab(sentences_list, progress_per=10000)
    model_ft.build_vocab(sentences_list, progress_per=10000)

    #train model
    model_w2v.train(sentences_list, total_examples=model_w2v.corpus_count, epochs=60, report_delay=1)
    model_ft.train(sentences_list, total_examples=model_ft.corpus_count, epochs=100, report_delay=1)

    # save model
    model_w2v.save(path_model_w2v)
    model_ft.save(path_model_ft)

    data_2 = data.copy(deep=True)

    # creating vectors
    data['text_vectors'] = data['text_words'].apply(lambda x: [get_avg_vector(sent, model) for sent in x])
    data['sentences'] = data.apply(lambda x: list(zip(x['text_sentences'], x['text_vectors'])), axis=1)

    data_2['text_vectors'] = data['text_words'].apply(lambda x: [get_avg_vector(sent, model_ft) for sent in x])
    data_2['sentences'] = data.apply(lambda x: list(zip(x['text_sentences'], x['text_vectors'])), axis=1)

    data_vectors = postprocess_vectors(data)
    data_vectors_2 = postprocess_vectors(data_2)

    data_vectors.to_pickle(path_dataset_vectors)
    data_vectors_2.to_pickle(path_dataset_vectors_2)

    time.sleep(50)
    default_prediction_filename = os.path.join(model_dir, "default_prediction.csv")
    pd.DataFrame(
        {
            "query": [""] * 5,
            "top_n": list(range(1, 6)),
            "text_page": [1] * 5,
            "doc_path": ["Valves_actuators/manuals-guides-betties.pdf"] * 5,
        }
    ).to_csv(default_prediction_filename)
    ###

#return text_json path
def download_train_data(model_dir="/home/model-server"):
    #model_dir = './'
    s3 = boto3.client("s3")
    zip_path = os.path.join(model_dir, "text_json.zip")

    s3.download_file("dsg-hackathon-dataset", "text_json.zip", zip_path)

    # unzip archive to '/home/model-server/text_json/'
    command_list = ["unzip", "-o", zip_path, "-d", model_dir]
    call(command_list)
    os.remove(zip_path)

    data_root = os.path.join(model_dir, "text_json")

    return data_root
