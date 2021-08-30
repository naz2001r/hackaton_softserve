import numpy as np

from utils.dataset import split_words


def cos_similarity(x1, x2):
    """Calculates cosine similarity between two vectors.

    Args:
        x1: vector 1
        x2: vector 2

    Returns:
        cos_sim: similarity coefficient
    """
    dot = np.dot(x1, x2)
    norma = np.sqrt(np.sum(np.square(x1)))
    normb = np.sqrt(np.sum(np.square(x2)))
    cos_sim = dot / (norma * normb)
    return cos_sim


def get_avg_vector_inference(text, model, nlp):
    """Converts text to vector.

    Args:
        text: input text
        model: mode
        nlp: spacy pipeline

    Returns:
        mean_vector: vector representation of text
    """
    words = split_words(text, nlp, 2)
    vector = [model.wv[word] for word in words if word in model.wv.vocab]
    if len(vector) == 0:
        print('Too many unknown words in request, output may be incorrect...')
        print()
        mean_vector = np.zeros(300)
        return mean_vector
    mean_vector = np.mean(vector, axis=0)
    return mean_vector


def get_doc_name(x):
    return '/'.join(x.split('/')[1:])


def get_files_subset(manufacturer, device_model, data):
    """Returns a subset of files related to the searched device.

    Args:
        manufacturer: device manufacturer
        device_model: device model
        data:

    Returns:
        files_subset: files subset
    """
    condition = (data['manufacturer'] == manufacturer) & \
                (data['device_model'].apply(lambda x: str(device_model) in x))
    files_subset = data[condition]['doc_name'].to_list()
    return files_subset


def search_similar_text(query_input, data_vec, data_names, vec_column, model, nlp, top):
    """Searches the top-K most similar text to a query.

    Args:
        query_input: search query
        data_vec: vectors dataset
        data_names: names dataset
        vec_column: vector column in vectors dataset
        model: model
        nlp: spacy pipeline
        top: number most similar responses

    Returns:
        df: DataFrame with search results
    """
    text = query_input['text']
    manufacturer = query_input['manufacturer']
    device_model = query_input['device_model']

    files_subset = get_files_subset(manufacturer, device_model, data_names)
    input_vector = get_avg_vector_inference(text, model, nlp)

    df = data_vec.copy()
    df['page_class_coordinate'] = df['page_class_coordinate'].apply(lambda x:
                                                                    [item for item in x if
                                                                     '_'.join(get_doc_name(item[0]).split('_')[:-1]) in files_subset])
    df = df[df['page_class_coordinate'].apply(len) > 0]
    df['similarity'] = df[vec_column].apply(lambda x: cos_similarity(input_vector, x))
    df = df[['text_sentences', 'text', 'page_class_coordinate', 'similarity']]
    df = df.sort_values(['similarity'], ascending=False).iloc[:top]
    if not files_subset:
        df['page_class_coordinate'] = [[(f'file_not_found.pdf_{i}', [])] for i in range(top)]
    return df
