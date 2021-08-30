import pandas as pd
import numpy as np


def get_avg_vector(words, model):
    """Converts a list of words to a vector.

    Args:
        words: input text in list of words format
        model: model

    Returns:
        mean_vector: vector representation of text
    """
    vector = [model.wv[word] for word in words if word in model.wv.vocab]
    if len(vector) == 0:
        mean_vector = np.zeros(300)
        return mean_vector
    mean_vector = np.mean(vector, axis=0)
    return mean_vector


def postprocess_vectors(data):
    """Prepares vectors dataset to inference format.

    Args:
        data: vectors dataset

    Returns:
        data_sent_grouped: processed vectors dataset
    """
    # group variable
    grouped_by = ['text_sentences', 'text']
    grouped_columns = ['doc_class', 'page_path', 'coordinate', 'text_sentences', 'text_vectors']

    # explode sentences column
    data_sent = data.explode('sentences').reset_index()
    print('There are {} rows in sentence DataFrame.'.format(data_sent.shape[0]))
    data_sent[['text_sentences', 'text_vectors']] = pd.DataFrame(data_sent['sentences'].to_list())

    # remove zero vectors
    data_sent = data_sent[data_sent['text_vectors'].apply(np.sum) != 0]
    print('There are {} rows in sentence DataFrame after removing zero vectors.'.format(data_sent.shape[0]))

    # convert to lower case text columns
    data_sent['text_sentences'] = data_sent['text_sentences'].str.lower().str.strip()
    data_sent['text'] = data_sent['text'].str.lower().str.strip()

    # group data based on sentence text and chunk text
    data_sent_grouped = data_sent.groupby(grouped_by)[grouped_columns].agg(lambda x: list(x)).reset_index()
    data_sent_grouped['text_vectors'] = data_sent_grouped['text_vectors'].apply(lambda x: x[0])
    data_sent_grouped['page_class_coordinate'] = data_sent_grouped.apply(lambda x:
                                                                         list(zip(x['page_path'], x['doc_class'],
                                                                                  x['coordinate'])), axis=1)
    # drop unused columns
    data_sent_grouped.drop(['page_path', 'doc_class', 'coordinate'], inplace=True, axis=1)
    print('There are {} rows in final FataFrame.'.format(data_sent_grouped.shape[0]))

    return data_sent_grouped
