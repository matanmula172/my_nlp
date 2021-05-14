import nltk
import pandas as pd
import numpy as np
import spacy
from gensim import corpora
from gensim.matutils import softcossim
from gensim.utils import simple_preprocess
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from transformers import AutoTokenizer
import gensim.downloader as api
import language_tool_python

nlp = spacy.load("en_core_web_sm")


def tokenize_text(text, is_stop_words=False, is_punctuation=False):
    doc = nlp(text)
    return [token for token in doc if token.is_stop == is_stop_words and token.is_punct == is_punctuation]


def get_sentences(text):
    return nltk.sent_tokenize(text)


# bert-base-uncased is a masked language model
# Masked Language Modeling is a fill-in-the-blank task, where a model
# uses the context words surrounding a mask token to try to predict what the masked word should be
# don't know why this model is the best for the encoding task
def encode_huggingface_transformer(text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer.encode(text)


def decode_huggingface_transformer(encoded_text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer.decode(encoded_text)


def remove_punctuations(text):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    return tokenizer.tokenize(text)


def stem_text(text):
    stemmer = PorterStemmer()
    stemmed_tokens = []
    tokenized_text = tokenize_text(text)
    for token in tokenized_text:
        stemmed_tokens.append(stemmer.stem(token))


# Lemmatization is the process of converting a word to its base form. The difference between stemming and
# lemmatization is, lemmatization considers the context and converts the word to its meaningful base form,
# whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.
def lemmatize_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def get_words_frequency(text, no_stop_words=True):
    # Removal of stop words and punctuations
    tokenized_text = tokenize_text(text, no_stop_words)

    freq_dict = dict()
    # Calculating frequency count
    for word in tokenized_text:
        key = str(word)
        if key not in freq_dict:
            freq_dict[key] = 1
        else:
            freq_dict[key] += 1
    return freq_dict


def get_n_most_frequent_words(text, n, is_stop_words=False):
    freq_dict = get_words_frequency(text, is_stop_words)
    sorted_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_dict[:n]


def text_correction(text):
    text = TextBlob(text)
    return text.correct()


def semantic_word_similarity(word1, word2):
    return nlp(word1).similarity(nlp(word2))


# get text similarity using cosine similarity
def text_similarity(text1, text2):
    return nlp(text1).similarity(nlp(text2))


# get similarity matrix using cosine similarity
def similarity_matrix(document_lst):
    # (sentence_index, feature_index) count
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(document_lst)

    # Obtaining the document-word matrix
    doc_term_matrix = matrix.todense()

    # Converting matrix to dataframe
    df = pd.DataFrame(doc_term_matrix)

    # Computing cosine similarity
    return cosine_similarity(df, df)


# Soft cosine similarity is similar to cosine similarity but in addition considers the semantic relationship between
# the words through its vector representation.
def soft_cosine_similarity(document_lst):
    # Prepare a dictionary and a corpus.
    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in document_lst])

    # load corpus to fit project
    fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    # Prepare the similarity matrix
    sim_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                     nonzero_limit=100)
    dictionary_vec = []
    for dic in document_lst:
        dictionary_vec.append(dictionary.doc2bow(simple_preprocess(dic)))

    len_array = np.arange(len(dictionary_vec))
    xx, yy = np.meshgrid(len_array, len_array)
    cosine_sim_mat = pd.DataFrame(
        [[round(softcossim(dictionary_vec[i], dictionary_vec[j], sim_matrix), 2) for i, j in zip(x, y)] for y, x in
         zip(xx, yy)])
    return cosine_sim_mat


# returns errors and how to correct them in a sentence
# error_id examples: UPPERCASE_SENTENCE_START, TOO_TO, EN_A_VS_AN, ENGLISH_WORD_REPEAT_RULE..
# https://predictivehacks.com/languagetool-grammar-and-spell-checker-in-python/#:~:text=LanguageTool%20is%20an%20open%2Dsource,through%20a%20command%2Dline%20interface.
# TODO: make this function faster
def language_checker(sentence):
    tool = language_tool_python.LanguageTool('en-US')
    return tool.check(sentence)
