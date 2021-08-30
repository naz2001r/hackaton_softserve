import json
import os

import pandas as pd


def get_page_path(doc_path, doc_class, page):
    """Creates page path based on document path, document class and page.

    Args:
        doc_path: document path
        doc_class: document class
        folder_pages: folder with pages

    Returns:
        page_path: page path
    """
    doc_name = doc_path.split('.')[0].split('/')[-1]
#     page_path = os.path.join(folder_pages, doc_class, f'{doc_name}.pdf', f'{doc_name}_{page}')
    page_path = os.path.join(doc_class, f'{doc_name}.pdf')
    page_path = f'{page_path}{str(page).replace("page", "")}'
    return page_path


def create_dataframe(file_path, doc_path_folder):
    """Converts text jsons to pandas DataFrame.

    Args:
        file_path: path to json file
        folder_pages: folder with pages

    Returns:
        df: dataframe with following columns:
        'doc_path', 'doc_class', 'page_path', 'page', 'chunk', 'text', 'coordinate
    """
    with open(file_path) as json_file:
        data = json.load(json_file)

    chunk_list = []
    for doc_path in data.keys():
        doc_path_plain = doc_path.replace(doc_path_folder, '')
        doc_class = doc_path_plain.split('/')[0]
        for page in data[doc_path].keys():
            page_path = get_page_path(doc_path_plain, doc_class, page)
            for nblock, block in enumerate(data[doc_path][page]):
                chunk_list.append({
                    'doc_path': doc_path_plain,
                    'doc_class': doc_class,
                    'page_path': page_path,
                    'page': page,
                    'chunk': 'chunk_{}'.format(nblock + 1),
                    'text': block[0],
                    'coordinate': block[1]
                })

    return pd.DataFrame(chunk_list)


def split_sentence(text, nlp, sentence_length):
    """Splits text into sentences.

    Args:
        text: input text
        nlp: spacy pipeline
        sentence_length: allowed number of characters in a sentence

    Returns:
        sentence_list: list of sentences
    """
    doc = nlp(text)
    sentence_list = [sent.text.strip()
                     for sent in doc.sents if len(sent.text.strip()) > sentence_length]
    return sentence_list


def split_words(text, nlp, word_length):
    """Splits text into a list of words.
    All stop words, non-characters words, and words with length less than 2 are removed

    Args:
        text: input text
        nlp: spacy pipeline
        word_length: allowed length of word in characters

    Returns:
        words: list of words (lower, lemma format)
    """
    words = [token.lemma_.lower()
             for token in nlp(text) if token.is_alpha and not token.is_stop and len(token) > word_length]
    return words
