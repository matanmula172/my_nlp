import nltk
from transformers import AutoTokenizer
import spacy


# nltk.download('punkt')
# nltk.download('stop')
# nltk.download('stopwords')
# nlp = spacy.load("en_core_web_sm")


def tokenize_text(text):
    return nltk.word_tokenize(text)


def get_sentences(text):
    return nltk.sent_tokenize(text)


# bert-base-uncased is a masked language model
# Masked Language Modeling is a fill-in-the-blank task, where a model
# uses the context words surrounding a mask token to try to predict what the masked word should be
# don't know why this model is the best for the encoding task
def encode_huggingface_transformer(text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer.encode(text)

# bla
def decode_huggingface_transformer(encoded_text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer.decode(encoded_text)
