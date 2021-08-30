"""
ModelHandler defines an example model handler for load and inference requests
"""
import json
import logging
import multiprocessing as mp
import os
import subprocess as sc
from spacy.lang.en import English
import pandas as pd
import numpy as np

from train import download_train_data, train_model
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from split_query import split_query
from prediction import search_similar_text


class ModelHandler(object):
    """
    A sample Model handler implementation.

    """

    def __init__(self):
        # mark state if model was initialized and loaded
        self.initialized = False
        # mark state that model training has been started
        self.training_started = False


        self.default_prediction = None
        self.model_w2v = None
        self.model_ft = None
        self.data_vectors = None
        self.data_vectors_2 = None
        self.data_names = None

        self.path_model_w2v = os.path.join("/home/model-server", 'word2vec.model')
        self.model_path_ft = os.path.join("/home/model-server", 'fasttext.model')
        self.path_dataset_vectors = os.path.join("/home/model-server", 'data_vectors.pkl')
        self.path_dataset_vectors_2 = os.path.join("/home/model-server", 'data_vectors_2.pkl')
        self.path_data_names = os.path.join("/home/model-server", 'device2document_map.pkl')

        # for test in local
        # self.model_path = os.path.join('./', 'word2vec.model')
        # self.path_dataset_vectors = os.path.join('./', 'data_vectors.pkl')
        # self.path_data_names = os.path.join('./', 'device2document_map.pkl')

        self.folder_list = [
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
        self.top_n = 5
        ###

    def initialize_model(self, model_dir="/home/model-server"):
        """
        Initialize model from model artifacts that were stored by a training process.
        """
        # for test in local
        # model_dir='./'




        print("listing dir in initialize model")
        print(os.listdir(model_dir))
        self.default_prediction = pd.read_csv(
            os.path.join(model_dir, "default_prediction.csv")
        )
        ###
        self.model_w2v = Word2Vec.load(self.model_path_w2v)
        self.model_ft = FastText.load(self.model_path_ft)


        self.data_vectors = pd.read_pickle(self.path_dataset_vectors)
        self.data_vectors_2 = pd.read_pickle(self.path_dataset_vectors_2)
        self.data_names = pd.read_pickle(self.path_data_names)

        # mark state that model was initialized successfully
        self.initialized = True
        print('Initialize correct')

    def predict(self, query: str, i) -> pd.DataFrame:
        def create_submission(predictions):
            predictions[['doc_path', 'text_page']] = predictions['page_class_coordinate'].apply(
                lambda x: pd.Series(x[0][0].split('.pdf_')))
            predictions['doc_path'] = predictions['doc_path'] + '.pdf'
            return predictions[['query', 'top_n', 'doc_path', 'text_page']]

        nlp_eng = English()
        sentencizer = nlp_eng.create_pipe("sentencizer")
        nlp_eng.add_pipe(sentencizer)

        query_input = query.iloc[i]
        query_text = query_input['manufacturer'] + ' ' + query_input['device_model'] + ' ' + query_input['text']
        data_output_w2v = search_similar_text(query_input, self.data_vectors, self.data_names,
                                          'text_vectors', self.model_w2v, nlp_eng, self.top_n)
        data_output_ft = search_similar_text(query_input, self.data_vectors_2, self.data_names,
                                             'text_vectors', self.model_ft, nlp_eng, self.top_n)
        data_output = pd.concat([data_output_w2v,data_output_ft]).sort_values(by=['similarity'],ascending=False)[:5]
        data_output['query'] = [query_text for _ in range(1, self.top_n + 1)]
        data_output['top_n'] = [i for i in range(1, self.top_n + 1)]
        submission_df = create_submission(data_output)
        return submission_df
        # raise NotImplementedError


    def inference(self, model_input):
        """
        Internal inference methods, runs model prediction for input queries.

        :param model_input: transformed model input data list
        :return: pd.DataFrame. Example:
                    'query': [''] * 5,
                    'top_n': [1,2,3,4,5],
                    'text_page': [10, 2, 12, 2, 34],
                    'doc_path': ['Valves_actuators/manuals-guides-betties.pdf',
                                'Valves_actuators/manuals-guides-betties.pdf',
                                'Valves_actuators/manuals-guides-betties.pdf',
                                'Valves_actuators/manuals-guides-betties.pdf',
                                'Valves_actuators/manuals-guides-betties.pdf']
        """

        #print(model_input)
        queries = pd.Series(model_input).apply(lambda x: split_query(x))
        queries.rename(columns={0: 'manufacturer', 1: 'device_model', 2: 'text'}, inplace=True)
        # queries = model_input
        # queries are in utf8, unconfigured logging.info may throw an error!
        responses = []
        # for query,text  in zip(queries.values, model_input):
        for i in range(len(model_input)):
            try:


                pred_df = self.predict(queries, i)

                ###
            except Exception as e:
                print(e)
                pred_df = self.default_prediction
            pred_df["query"] = [model_input[i] for _ in range(self.top_n)]
            if 'Unnamed: 0' in pred_df.columns:
                pred_df = pred_df.drop('Unnamed: 0', axis=1)

            # print(pred_df['query'])
            pred_df.text_page = pred_df.text_page.apply(lambda x: int(x))
            responses.append(pred_df.to_json())

        #print(responses)
        return responses

    def start_training(self, model_dir="/home/model-server"):
        """
        Method downloads training data and starts model training in another process.
        """
        # Load data for training
        data_root = download_train_data(model_dir)

        # for test in local
        # data_root = 'DS-repo/baseline/text_json'

        # start training your model and store model artifact to '/home/model-server'
        mp.Process(target=train_model, args=(model_dir, data_root)).start()

        # set flag after training started!
        self.training_started = True

    def preprocess(self, request):
        """
        Take the input data and pre-process it make it inference ready.

        :param request: list of raw requests
        :return: list of strings
        """
        queries = []
        for idx, data in enumerate(request):
            query = data.get("body")
            query = query.decode("utf-8")

            queries.append(query)

        return queries

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        model_input = self.preprocess(data)
        # model_input = data
        model_out = self.inference(model_input)
        return model_out


_service = ModelHandler()


def handle(data, context):
    # start training on first request to endpoint
    if not _service.training_started:
        _service.start_training()
        return [
            json.dumps(
                {
                    "status": 200,
                    "message": "training started, please wait till making next request",
                }
            )
        ]
    # try to load model
    print('error')
    if not _service.initialized:
        try:
            _service.initialize_model()
            return [
                json.dumps({"status": 200, "message": "model initialized successfull"})
            ]
        except Exception as e:
            print('error')
            logging.error(e)
            return [
                json.dumps(
                    {
                        "status": 503,
                        "message": "attempt to initialize model failed, \
                        please wait till the model is trained and retry again",
                    }
                )
            ]
    if data is None:
        return [json.dumps({"status": 404, "message": "data is None"})]
    return _service.handle(data, context)

# data=pd.read_csv('baseline/validation.csv')['query'].unique()[:5]
# print(data)
# _service.start_training()
# handle(data, 0)
# _service.initialize_model()
# _service.inference(data)
