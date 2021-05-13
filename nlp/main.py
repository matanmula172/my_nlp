import nltk
from transformers import AutoTokenizer
from nltk.stem import PorterStemmer
import spacy
from textblob import TextBlob

# nltk.download('punkt')
# nltk.download('stop')
# nltk.download('stopwords')
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


def text_similarity(text1, text2):
    return nlp(text1).similarity(nlp(text2))
