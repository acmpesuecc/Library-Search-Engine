import pickle
import os
from flask import Flask, render_template, request, flash
import pandas as pd
import spacy
import string
import gensim
import operator
import re
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
import unicodedata

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Load dataset
dataset = "Book_Dataset_1.csv"
df_books = pd.read_csv(dataset)

# Remove unnecessary columns
columns_to_remove = ['Price', 'Price_After_Tax', 'Tax_amount', 'Avilability', 'Number_of_reviews']
df_books = df_books.drop(columns=columns_to_remove)

# Load stop words
spacy_nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

def spacy_tokenizer(sentence):
    # Normalize to NFC - handle non-ASCII characters better
    sentence = unicodedata.normalize("NFC", sentence)
    
    # Optimized regex patterns
    sentence = re.sub(r"[‘’`]", "'", sentence) 
    sentence = re.sub(r"\w*\d\w*", "", sentence) 
    sentence = re.sub(r" +", " ", sentence.strip())  
    sentence = re.sub(r"\n+", " ", sentence) 
    sentence = re.sub(r"[^\w\s.,!?]", " ", sentence) 
    
    tokens = spacy_nlp(sentence)
    tokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ 
        for word in tokens
    ]
    
    tokens = [
        word for word in tokens 
        if word not in stop_words 
        and word not in punctuations 
        and len(word) > 2 
        and not word.isspace()
    ]
    
    return tokens

# Create tokenized description column
df_books['Book_Description_tokenized'] = df_books['Book_Description'].map(lambda x: spacy_tokenizer(x))

# Load pre-trained models or train models if necessary
models_path = 'models.pickle'
try:
    if os.path.exists(models_path):
        with open(models_path, 'rb') as f:
            book_tfidf_model, book_lsi_model, dictionary = pickle.load(f)
    else:
        raise FileNotFoundError("The model file does not exist.")
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    flash(f"Error loading models: {str(e)}. Please ensure the 'models.pickle' file is present and not corrupted. You may need to retrain the models.")
    # Call your model training function here if needed
    # train_models()  # Uncomment and implement this function as needed

    # Create and train TF-IDF model
    dictionary = corpora.Dictionary(df_books['Book_Description_tokenized'])
    corpus = [dictionary.doc2bow(desc) for desc in df_books['Book_Description_tokenized']]
    book_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)

    # Create and train LSI model
    book_lsi_model = gensim.models.LsiModel(book_tfidf_model[corpus], id2word=dictionary, num_topics=300)

    # Save models to pickle file
    with open(models_path, 'wb') as f:
        pickle.dump((book_tfidf_model, book_lsi_model, dictionary), f)

# Load indexed corpus
book_tfidf_corpus = gensim.corpora.MmCorpus('book_tfidf_model_mm')
book_lsi_corpus = gensim.corpora.MmCorpus('book_lsi_model_mm')
book_index = MatrixSimilarity(book_lsi_corpus, num_features=book_lsi_corpus.num_terms)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_similar_books(query, dictionary)
    return render_templat
