import pandas as pd


def preparing_data_w2v(data):
    """Prepares data for Word2vec model.

    Args:
        data: pandas DataFrame with text column

    Returns:
        sentences_clean_list: a list of sentences is prepared for Word2vec model
    """
    # remove duplicates in `text` column
    df_text = data.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
    print('There are {} rows after removing duplicates.'.format(df_text.shape[0]))
    # convert sentences column to list of sentences
    sentences_list = [sent for block in df_text['text_words'].to_list() for sent in block]
    print('{} sentences were created.'.format(len(sentences_list)))
    # create dataset from sentences
    df_sent = pd.DataFrame(sentences_list)
    # remove duplicates in sentences
    df_sent = df_sent.drop_duplicates(keep='first').reset_index(drop=True)
    sentences_clean_list = df_sent.values.tolist()
    sentences_clean_list = [[word for word in sent if isinstance(word, str)] for sent in sentences_clean_list]
    print('{} sentences were created after filtering duplicated sentences.'.format(len(sentences_clean_list)))
    return sentences_clean_list
