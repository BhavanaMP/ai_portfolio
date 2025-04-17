import warnings
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
from keybert import KeyBERT
from sklearn.preprocessing import StandardScaler
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import nltk
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import hstack
import os
import pickle
import random
import string
import pandas as pd
from scipy.sparse import hstack

warnings.filterwarnings('ignore')


class LDAVectorizerPreFit():
    lda_model = None
    lda_dictionary = None
    total_docs = None

    def __init__(self, lda_model, lda_dictionary, total_docs):
        super().__init__()
        self.lda_model = lda_model
        self.lda_dictionary = lda_dictionary
        self.total_docs = total_docs

    @staticmethod
    def sent_to_words(sentences):
        for sentence in sentences:
            yield (simple_preprocess(str(sentence), deacc=True))

    @staticmethod
    def get_corpus(data):
        """
        Get LDA Model, Corpus, id2word mapping
        """

        tokenized_docs = list(LDAVectorizerPreFit.sent_to_words(data))

        id2word = Dictionary(tokenized_docs)
        corpus = [id2word.doc2bow(text) for text in tokenized_docs]
        return corpus, id2word, tokenized_docs
  
    def lda_topics(self, corpus):
        train_vecs = []
        rootdir = os.getcwd()
        corpus_pickle = None
        corpus_lda_pkl = os.path.join(rootdir, 'corpus_lda.pkl')
        if os.path.isfile(corpus_lda_pkl) and os.path.exists(corpus_lda_pkl):
            with open(corpus_lda_pkl, 'rb') as f:
                corpus_pickle = pickle.load(f)
        for i in range(self.total_docs):
            if corpus_pickle:
                top_topics = (self.lda_model.get_document_topics(
                    corpus_pickle[i], minimum_probability=0.0
                ))
            else:
                top_topics = (self.lda_model.get_document_topics(
                    corpus[i], minimum_probability=0.0
                ))
            topic_vec = [top_topics[i][1] for i in range(len(top_topics))]
            train_vecs.append(topic_vec)
        return train_vecs


MAX_FEATURES = 2500
CHUNKS = 500


class FeatureExtractor():
    tfidf_model = lda_model = w2v_model = kp_model = None
    tfdif_vector = lda_vector = w2v_vector = kp_vector = None
    lda_dictionary = key_phrases = None
    dataset = transformed_stacked_vector = cluster_labels = None

    def extract_features(self, data_path, is_new_data):
        print("Downloading NLTK models...")
        FeatureExtractor.download_nltk_models()

        print("Extracting dataset...")
        parsed_docs = FeatureExtractor.download_dataset(data_path)

        print("No. of docs: ", len(parsed_docs))

        print("Preparing the dataset...")
        processed_data = FeatureExtractor.prepare_data(parsed_docs)

        lemmatized_data = FeatureExtractor.lemmatize_data(processed_data)

        self.dataset = lemmatized_data

        print("Training the models...")
        tfidf_pretrained_vect, pretrained_tfidf_model,\
            lda_pretrained_vect, pretrained_lda_model, lda_dictionary,\
            vect_pretrained_word2vec, pretrained_word2vec_model,\
            pretrained_kp_model, key_phrases =\
            FeatureExtractor.train_vectorizer_models(lemmatized_data)

        self.tfidf_model = pretrained_tfidf_model
        self.lda_model = pretrained_lda_model
        self.w2v_model = pretrained_word2vec_model
        self.kp_model = pretrained_kp_model
        self.lda_vector = lda_pretrained_vect
        self.w2v_vector = vect_pretrained_word2vec
        self.tfdif_vector = tfidf_pretrained_vect
        self.key_phrases = key_phrases
        self.lda_dictionary = lda_dictionary

        print("Transformation, Truncated SVD and hstack of data...")
        transformed_stacked_vector =\
            FeatureExtractor.create_feature_hstack_with_svd(
                self.tfdif_vector, self.lda_vector, self.w2v_vector
            )
        self.transformed_stacked_vector = transformed_stacked_vector

        print("Done!")
        return transformed_stacked_vector

    @staticmethod
    def download_nltk_models():
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        nltk.download('wordnet')

    @staticmethod
    def download_dataset(datapath):
        rootdir = os.getcwd()
        parsed_pkl = os.path.join(rootdir, 'parsed_doc.pkl')
        if os.path.isfile(parsed_pkl) and os.path.exists(parsed_pkl):
            with open(parsed_pkl, 'rb') as f:
                parsed_docs = pickle.load(f)
                return parsed_docs
        iterable_lines = []
        for subdir, dirs, files in os.walk(datapath):
            for file in files:
                with open((str(subdir)+"/"+str(file)), 'r',
                          encoding="ISO-8859-1") as f:
                    lines = f.read()
                    iterable_lines += [lines]
        random.shuffle(iterable_lines)
        return iterable_lines

    @staticmethod
    def prepare_data(parsed_docs):
        rootdir = os.getcwd()
        preprocessed_doc_pkl = os.path.join(rootdir, 'preprocessed_doc.pkl')
        if os.path.isfile(preprocessed_doc_pkl) and\
           os.path.exists(preprocessed_doc_pkl):
            with open(preprocessed_doc_pkl, 'rb') as f:
                preprocessed_articles = pickle.load(f)
                return preprocessed_articles
        processed_articles = []
        for i in range(len(parsed_docs)):
            document = parsed_docs[i]
            processed_article = document.lower()
            processed_article = re.sub('\[.*?\]', '', processed_article)
            processed_article = re.sub('https?://\S+|www\.\S+', '',
                                       processed_article)
            processed_article = re.sub('<.*?>+', '', processed_article)
            processed_article = re.sub('[%s]' % re.escape(string.punctuation),
                                       '', processed_article)
            processed_article = re.sub('\n', '', processed_article)
            processed_article = re.sub('\w*\d\w*', '', processed_article)
            processed_article = processed_article.lstrip(' ')
            processed_article = "".join([i for i in processed_article
                                         if i not in string.punctuation])
            processed_articles.append(processed_article)
        return processed_articles

    @staticmethod
    def lemmatize_data(processed_doc):
        rootdir = os.getcwd()
        preprocessed_doc_pkl = os.path.join(rootdir, 'wo_stopwords.pkl')
        if os.path.isfile(preprocessed_doc_pkl) and\
           os.path.exists(preprocessed_doc_pkl):
            with open(preprocessed_doc_pkl, 'rb') as f:
                preprocessed_articles = pickle.load(f)
                return preprocessed_articles
        stopword_set = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        lemmatized_docs = []
        for i in range(len(processed_doc)):
            word_list = nltk.word_tokenize(processed_doc[i])
            wordlist = [word for word in word_list
                        if word not in stopword_set]
            lemmatized_output = ' '.join([lemmatizer.lemmatize(w)
                                          for w in wordlist])
            lemmatized_docs.append(lemmatized_output)

        return lemmatized_docs

    @staticmethod
    def train_LDA_model(lemmatized_docs):
        train_corpus, train_id2word, tokenized_docs =\
            LDAVectorizerPreFit.get_corpus(lemmatized_docs)
        total_docs = len(tokenized_docs)
        rootdir = os.getcwd()
        pretrained_lda_model_pkl = os.path.join(rootdir, 'lda_model.pkl')
        if os.path.isfile(pretrained_lda_model_pkl) and\
           os.path.exists(pretrained_lda_model_pkl):
            with open(pretrained_lda_model_pkl, 'rb') as f:
                pretrained_lda_model = pickle.load(f)
                return pretrained_lda_model, train_id2word, total_docs

        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=train_corpus, id2word=train_id2word,
            num_topics=20, random_state=100, update_every=1,
            chunksize=2000, passes=10, iterations=10, alpha='auto',
            eval_every=1, per_word_topics=True
        )
        return lda_model, train_id2word, total_docs

    @staticmethod
    def train_word2vect_model(lemmatized_docs):
        rootdir = os.getcwd()
        pretrained_w2v_model_pkl = os.path.join(rootdir, 'w2v_model.pkl')
        if os.path.isfile(pretrained_w2v_model_pkl) and\
           os.path.exists(pretrained_w2v_model_pkl):
            with open(pretrained_w2v_model_pkl, 'rb') as f:
                word2vec_model = pickle.load(f)
                return word2vec_model

        all_words = [nltk.word_tokenize(sentence)
                     for sentence in lemmatized_docs]

        word2vec_model = Word2Vec(all_words, window=3, min_count=5, workers=4)

        return word2vec_model

    @staticmethod
    def train_Key_phrase_model(lemmatized_docs):
        rootdir = os.getcwd()
        kp_model = KeyBERT()
        keyphrases_pkl = os.path.join(rootdir, 'wo_lemm_kp.pkl')
        if os.path.isfile(keyphrases_pkl) and\
           os.path.exists(keyphrases_pkl):
            with open(keyphrases_pkl, 'rb') as f:
                keyphrases = pickle.load(f)
                return kp_model, keyphrases
        kp_model = keyphrases = None
        # Commented this block as the model is already ran and we have pickled
        # split_docs = (lemmatized_docs[i:i+CHUNKS]
        #               for i in range(0, len(lemmatized_docs), CHUNKS))
        # keyphrases = []
        # for spl in split_docs:
        #     keyphrases += kp_model.extract_keywords(docs=spl)
        return kp_model, keyphrases

    @staticmethod
    def train_tfidf_model(lemmatized_docs):
        rootdir = os.getcwd()
        tfidf_vectorizer_pkl = os.path.join(rootdir, 'tfidf_vectorizer_2322.pkl')
        tfidf_vector_pkl = os.path.join(rootdir, 'vect_tfidf_2322.pkl')
        if os.path.isfile(tfidf_vectorizer_pkl) and\
           os.path.exists(tfidf_vectorizer_pkl):
            with open(tfidf_vectorizer_pkl, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
                if os.path.isfile(tfidf_vector_pkl) and\
                   os.path.exists(tfidf_vector_pkl):
                    with open(tfidf_vector_pkl, 'rb') as f:
                        tfidf_vector = pickle.load(f)
                        return tfidf_vectorizer, tfidf_vector

        tfidf_vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES, sublinear_tf=True,
            max_df=1, stop_words="english", ngram_range=(3, 3)
        )
        tfidf_vector = tfidf_vectorizer.fit_transform(lemmatized_docs)
        return tfidf_vectorizer, tfidf_vector

    @staticmethod
    def train_vectorizer_models(X):
   
        print("Training TF-IDF model")
        pretrained_tfidf_model, tfidf_pretrained_vect =\
            FeatureExtractor.train_tfidf_model(X)

        print("Training LDA model...")
        pretrained_lda_model, lda_dictionary, total_docs =\
            FeatureExtractor.train_LDA_model(X)
        vect_pretrained_obj = LDAVectorizerPreFit(pretrained_lda_model,
                                                  lda_dictionary,
                                                  total_docs)
        lda_vect = vect_pretrained_obj.lda_topics(lda_dictionary)
        scaler = StandardScaler()
        lda_pretrained_vect = scaler.fit_transform(np.array(lda_vect))

        print("Training Word2vec model...")
        pretrained_word2vec_model = FeatureExtractor.train_word2vect_model(X)
        vocab_w2v = pretrained_word2vec_model.wv[
            pretrained_word2vec_model.wv.index_to_key
        ]
        vect_pretrained_word2vec = vocab_w2v.transpose()
        
        print("Training Key Phrase model...")
        pretrained_kp_model, key_phrases =\
            FeatureExtractor.train_Key_phrase_model(X)

        return tfidf_pretrained_vect, pretrained_tfidf_model,\
            lda_pretrained_vect, pretrained_lda_model, lda_dictionary,\
            vect_pretrained_word2vec, pretrained_word2vec_model,\
            pretrained_kp_model, key_phrases

    @staticmethod
    def create_feature_hstack_with_svd(
        tfidf_vectorizer, lda_vectorizer, w2v_vectorizer
    ):
        rootdir = os.getcwd()
        transformed_stacked_pkl = os.path.join(
            rootdir, 'stacked_total_final_reduced_200.pkl'
        )
        if os.path.isfile(transformed_stacked_pkl) and\
           os.path.exists(transformed_stacked_pkl):
            with open(transformed_stacked_pkl, 'rb') as f:
                transformed_stacked_data = pickle.load(f)
                return transformed_stacked_data

        print("Combining stacks...")
        stacked_data = hstack([
            tfidf_vectorizer, lda_vectorizer, w2v_vectorizer
        ])
        print("Printing combined stack size...")
        print(stacked_data.shape)
        clf = TruncatedSVD(MAX_FEATURES)
        transformed_stacked_data = clf.fit_transform(stacked_data)
        print("Printing transformed stack size...")
        print(transformed_stacked_data.shape)
        return transformed_stacked_data
