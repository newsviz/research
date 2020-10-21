import pandas as pd
import ast
import artm
import matplotlib.pyplot as plt
import os
import pymorphy2
import re
import sys
import numpy as np
import pprint
import shutil
import tqdm

from logging_functions import init_logger

# TODO should be import from package not with using sys
sys.path.append('../../../preprocessing')
from preprocessing_tools import clean_text, lemmatization


class TopicModel:
    def __init__(self, log_params=None):
        self.word_cache = {}
        self.morph = pymorphy2.MorphAnalyzer()

        if log_params is None:
            log_params = {}

        logname = log_params.pop('logger_name', 'Topic model')
        logpath = log_params.pop('logpath', 'logs/bigartm.log')
        logging_stdout = log_params.pop('logging_stdout', True)
        self.logger = init_logger(logname, logpath, logging_stdout)

        # initialized in methods
        self.batch_vectorizer = None
        self.dictionary = None

    def load_data(self, path, convert_processed_text=False):
        self.logger.info("Loading data from %s ...", path)
        df = pd.read_csv(path)
        if convert_processed_text:
            df['lemmatized_text'] = df['lemmatized_text'].apply(
                ast.literal_eval)
        return df

    @staticmethod
    def save_data(data, path, **kwargs):
        dirpath = os.path.dirname(path)
        os.makedirs(dirpath, exist_ok=True)
        data.to_csv(path, **kwargs)

    @staticmethod
    def get_date(data, column, regexp=None, date_fmt=None):
        regexp = regexp or re.compile(r"(\d{4}/\d{2}/\d{2})")
        date = data[column].str.extract(regexp, expand=False)
        return pd.to_datetime(date, format=date_fmt)

    def normalize_word(self, word):
        """Normalize with caching."""
        if word not in self.word_cache:
            self.word_cache[word] = self.morph.parse(word)[0].normal_form
        return self.word_cache[word]

    def tokenize(self, text, re_token=None):
        """Strip text to words then normalize it."""
        if not re_token:
            re_token = re.compile(r'[\'\w]+')

        text = re_token.findall(text.lower())
        return text

    def tokenize_normalize(self, data_text, use_preprocessing=True):
        self.logger.info("Tokenize and normalize text...")
        if use_preprocessing:
            lemmatized_text = data_text.apply(
                lambda x: lemmatization(clean_text(x)))
        else:
            lemmatized_text = data_text.apply(
                lambda x: [self.normalize_word(word)
                           for word in self.tokenize(x)])
        return lemmatized_text

    @staticmethod
    def get_docid(data):
        return list(range(data.shape[0]))

    def prepare_data_for_model(self, data=None, vwpath='data/lenta.vw',
                               batches_path='data/batches'):
        create_batch_files = False if data is None else True
        if create_batch_files:
            self._create_vw_file(data, vwpath=vwpath)

        self._init_batch_vectorizer(vwpath=vwpath, batches_path=batches_path,
                                    create_batch_files=create_batch_files)
        self._gather_dictionary(batches_path=batches_path)

    def _create_vw_file(self, data, vwpath='lenta.vw'):
        self.logger.info("Creating VW file %s ...", vwpath)
        with open(vwpath, 'w') as fp:
            for text, did in data[['lemmatized_text', 'docID']].values:
                fp.write('{} | {}\n'.format(did, ' '.join(text)))

    def _init_batch_vectorizer(self, vwpath='data/lenta.vw',
                               batches_path='data/batches',
                               create_batch_files=True):
        self.logger.info("Init BatchVectorizer for %s ...", batches_path)
        if create_batch_files:
            batch_vect = artm.BatchVectorizer(data_path=vwpath,
                                              data_format='vowpal_wabbit',
                                              target_folder=batches_path)
        else:
            self.logger.info("Create from existing batches")
            batch_vect = artm.BatchVectorizer(data_path=batches_path,
                                              data_format='batches')
        self.batch_vectorizer = batch_vect

    def _gather_dictionary(self, batches_path='data/batches'):
        self.logger.info("Gather dictionary from %s ...", batches_path)
        self.dictionary = artm.Dictionary()
        self.dictionary.gather(data_path=batches_path)

    def filter_dictionary(self, **kwargs):
        self.dictionary.filter(**kwargs)

    def init_model(self, model_level=None, model_name='ARTM', num_topics=100,
                   **kwargs):
        if model_name not in ('ARTM', 'LDA', 'hARTM'):
            raise ValueError(f'No model in artm for {model_name}')

        self.logger.info("Initializing %s model", model_name)
        if model_name == 'hARTM' and model_level is not None:
            model = self._init_hartm(model_level)
        else:
            model_type = getattr(artm, model_name)
            model = model_type(num_topics=num_topics,
                               dictionary=self.dictionary, **kwargs)
        self.model = model

    def _init_hartm(self, model_level):
        model_level.initialize(dictionary=self.dictionary)
        return model_level

    def add_scores(self, log_scores=True, **kwargs):
        for scorer, values in kwargs.items():
            scorer_name = values.pop('name', scorer)
            overwrite = values.pop('overwrite', False)
            self.model.scores.add(getattr(artm, scorer)(
                name=scorer_name, **values), overwrite=overwrite)

            if not log_scores:
                continue

            if not values:
                self.logger.info("Scorer %s is added to the model",
                                 scorer_name)
            else:
                self.logger.info(
                    "Scorer %s is added to the model with arguments:\n%s",
                    scorer_name, pprint.pformat(values))

    def add_regularizers(self, log_regularizers=True, **kwargs):
        for reg, values in kwargs.items():
            reg_name = values.pop('name', reg)
            overwrite = values.pop('overwrite', False)
            self.model.regularizers.add(getattr(artm, reg)(
                name=reg_name), overwrite=overwrite)

            if not values:
                if log_regularizers:
                    self.logger.info("Regularizer %s is added to the model.")
                continue

            if isinstance(values, dict):
                for key, val in values.items():
                    setattr(self.model.regularizers[reg_name], key, val)
                if log_regularizers:
                    self.logger.info(
                        "Regularizer %s is added to the model " +
                        "with arguments:\n%s", reg_name,
                        pprint.pformat(values))
            else:
                raise ValueError("Pass regularizer arguments as dict")

    def fit_model(self, method='offline', num_collection_passes=10, **kwargs):
        if method == 'offline':
            self.model.fit_offline(
                batch_vectorizer=self.batch_vectorizer,
                num_collection_passes=num_collection_passes,
                **kwargs)
        elif method == 'online':
            self.model.fit_online(**kwargs)

    def save_model(self, path='model_fitted', force=False):
        if os.path.exists(path) and force:
            shutil.rmtree(path)

        self.model.dump_artm_model(path)

    def load_model(self, path='model_fitted'):
        self.model = artm.load_artm_model(path)

    def get_time_topics(self, id_date, topics=None):
        theta = self.model.transform(batch_vectorizer=self.batch_vectorizer)
        theta = theta.T
        joined = id_date.join(theta)
        if topics is None:
            topics = ['topic_{}'.format(i) 
                      for i in range(self.model.num_topics)]
        gb = joined.groupby(['year', 'month'])[topics].sum()
        return gb

    def print_measures(self):
        for score_name, scorer in self.model.score_tracker.items():
            if hasattr(scorer, 'last_value'):
                print(f'{score_name}: {scorer.last_value: .3f}')
            elif hasattr(scorer, 'last_average_coherence'):
                print(f'{score_name}: {scorer.last_average_coherence: .3f}')

    def plot_perplexity(self, skip_first=True, score_name='PerplexityScore'):
        return self.plot_score(score_name, skip_first=skip_first)

    def plot_score(self, score_name, skip_first=True, ylabel=None):
        idx_start = 1 if skip_first else 0
        plt.plot(list(range(self.model.num_phi_updates))[idx_start:],
                 self.model.score_tracker[score_name].value[idx_start:],
                 'r--', linewidth=2)
        if ylabel is None:
            ylabel = 'ARTM perp. (red)'
        plt.xlabel('Iterations count')
        plt.ylabel(ylabel)
        plt.grid(True)
        return plt.gca()


    @staticmethod
    def get_id_date(data, id_col='docID'):
        return data[[id_col, 'year', 'month']]


def tune_topics(tm):
    perplexity_values_topics = {}
    num_topics_range = [10, 50, 100, 200, 500]
    for num_topics in tqdm.tqdm(num_topics_range):
        tm.init_model(num_topics=num_topics)

        scores = ['PerplexityScore', 'SparsityThetaScore',
                  'SparsityPhiScore', 'TopTokensScore', 'TopicKernelScore']
        kwarg_scores = {scorer: {'name': scorer} for scorer in scores}
        tm.add_scores(**kwarg_scores)

        regularizers = {
            'SmoothSparsePhiRegularizer': {'tau': -1.0},
            'SmoothSparseThetaRegularizer': {'tau': -0.5},
            'DecorrelatorPhiRegularizer': {'tau': 1e5}
        }
        tm.add_regularizers(**regularizers)

        tm.fit_model(num_collection_passes=10)
        perplexity_values_topics[num_topics] = \
            tm.model.score_tracker['PerplexityScore'].value
        print(f'Num topics: {num_topics}')
        tm.print_measures()
    return pd.DataFrame(perplexity_values_topics)


def tune_tau_decorrelator(tm, num_topics=200):
    perplexity_values_dec_tau = {}
    decorrelator_phi_tau_space = np.geomspace(0.1, 1e7, num=9)
    for tau in tqdm.tqdm(decorrelator_phi_tau_space):
        tm.init_model(num_topics=num_topics)

        scores = ['PerplexityScore', 'SparsityThetaScore',
                  'SparsityPhiScore', 'TopTokensScore', 'TopicKernelScore']
        kwarg_scores = {scorer: {'name': scorer} for scorer in scores}
        tm.add_scores(**kwarg_scores)

        regularizers = {
            'SmoothSparsePhiRegularizer': {'tau': -1.0},
            'SmoothSparseThetaRegularizer': {'tau': -0.5},
            'DecorrelatorPhiRegularizer': {'tau': tau}
        }
        tm.add_regularizers(**regularizers)

        tm.fit_model(num_collection_passes=10)
        perplexity_values_dec_tau[tau] = \
            tm.model.score_tracker['PerplexityScore'].value
        print(f'Num topics: {num_topics}')
        tm.print_measures()
    return pd.DataFrame(perplexity_values_dec_tau)


def main(full_processing=True,
         save_processing_data=True,
         lenta_dataset_path='../../data/datasets/news_lenta.csv',
         processed_dataset_path='processed_data/data_processed.csv',
         model_path='model_fitted',
         tune_model=False):

    tm = TopicModel()

    if full_processing:
        data = tm.load_data(lenta_dataset_path)

        date = tm.get_date(data, column='url')
        data['year'] = date.dt.year
        data['month'] = date.dt.month

        data['docID'] = tm.get_docid(data)
        data['lemmatized_text'] = tm.tokenize_normalize(data['text'])

        if save_processing_data:
            tm.save_data(data, processed_dataset_path)
    else:
        data = tm.load_data(processed_dataset_path)

    id_date = tm.get_id_date(data)

    tm.prepare_data_for_model(data=None)
    tm.filter_dictionary(min_tf=10, max_df_rate=0.1)
    tm.init_model(num_topics=200)

    scores = ['PerplexityScore', 'SparsityThetaScore',
              'SparsityPhiScore', 'TopTokensScore', 'TopicKernelScore']
    kwarg_scores = {scorer: {'name': scorer} for scorer in scores}
    tm.add_scores(**kwarg_scores)

    regularizers = {
        'SmoothSparsePhiRegularizer': {'tau': -1.0},
        'SmoothSparseThetaRegularizer': {'tau': -0.5},
        'DecorrelatorPhiRegularizer': {'tau': 1e5}
    }
    tm.add_regularizers(**regularizers)

    tm.fit_model(num_collection_passes=10)
    tm.save_model(path=model_path, force=True)

    tm.print_measures()
    tm.plot_perplexity()

    sparse_scores = {}
    sparse_scores['SparsityPhiScore'] = \
        tm.model.score_tracker['SparsityPhiScore'].value
    sparse_scores['SparsityThetaScore'] = \
        tm.model.score_tracker['SparsityThetaScore'].value
    pd.DataFrame(sparse_scores).plot(grid=True)

    top_tokens = tm.model.score_tracker['TopTokensScore']
    for topic in list(top_tokens.last_tokens.keys())[:20]:
        print(topic, *top_tokens.last_tokens[topic])

    phi = tm.model.get_phi()
    phi['word'] = phi.index

    for col in phi.columns[:10]:
        if col != 'word':
            print(col)
            print(phi[[col, 'word']].sort_values(
                by=col, ascending=False)['word'].values[:20])

    gb = tm.get_time_topics(id_date)

    if tune_model:
        df_topics = tune_topics(tm)
        df_topics.iloc[1:].plot()

        df_topics_tau_decor = tune_tau_decorrelator(tm)
        df_topics_tau_decor.iloc[1:].plot()

    return gb


if __name__ == '__main__':
    gb = main()
