import pickle
from flask import Flask, render_template, request
import pandas as pd
import spacy
import string
import gensim
import operator
import re
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter

app = Flask(__name__)

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
    # Data cleaning
    sentence = re.sub('\'','',sentence)
    sentence = re.sub(r'\w*\d\w*','',sentence)
    sentence = re.sub(' +',' ',sentence)
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    sentence = re.sub(r'\n',' ',sentence)
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    # Tokenization, lemmatization, stop words removal
    tokens = spacy_nlp(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    return tokens

# Create tokenized description column
df_books['Book_Description_tokenized'] = df_books['Book_Description'].map(lambda x: spacy_tokenizer(x))

# Load pre-trained models or train models if necessary
try:
    with open('models.pickle', 'rb') as f:
        book_tfidf_model, book_lsi_model, dictionary = pickle.load(f)
except FileNotFoundError:
    # Create and train TF-IDF model
    dictionary = corpora.Dictionary(df_books['Book_Description_tokenized'])
    corpus = [dictionary.doc2bow(desc) for desc in df_books['Book_Description_tokenized']]
    book_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)

    # Create and train LSI model
    book_lsi_model = gensim.models.LsiModel(book_tfidf_model[corpus], id2word=dictionary, num_topics=300)

    # Save models to pickle file
    with open('models.pickle', 'wb') as f:
        pickle.dump((book_tfidf_model, book_lsi_model, dictionary), f)


# Load indexed corpus
book_tfidf_corpus = gensim.corpora.MmCorpus('book_tfidf_model_mm')
book_lsi_corpus = gensim.corpora.MmCorpus('book_lsi_model_mm')
book_index = MatrixSimilarity(book_lsi_corpus, num_features = book_lsi_corpus.num_terms)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_similar_books(query, dictionary)
    return render_template('results.html', results=results)

def search_similar_books(search_term, dictionary):
    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = book_tfidf_model[query_bow]
    query_lsi = book_lsi_model[query_tfidf]

    book_index.num_best = 5

    books_list = book_index[query_lsi]

    books_list.sort(key=itemgetter(1), reverse=True)
    book_names = []

    for j, book in enumerate(books_list):
        # Truncate the book description to the first three sentences
        description = df_books['Book_Description'][book[0]]
        sentences = re.split(r'(?<=[.!?])\s+', description)[:3]  # Split sentences
        truncated_description = ' '.join(sentences)

        book_names.append({
            'Relevance': round((book[1] * 100),2),
            'book Title': df_books['Title'][book[0]],
            'book Plot': truncated_description,
            'Image_Link': df_books['Image_Link'][book[0]]
        })

        if j == (book_index.num_best-1):
            break

    return book_names

if __name__ == '__main__':
    app.run(debug=True)
