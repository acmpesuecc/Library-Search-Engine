# Library-Search-Engine

This project is a web-based application that recommends books based on a user's query. The app utilizes NLP techniques to tokenize and analyze book descriptions, and applies TF-IDF and Latent Semantic Indexing (LSI) models to find and recommend similar books. Flask is used for the backend server and web interface.

**Note:** If you're a contributer please read CONTRIBUTING.md

## Features

-  Book Similarity Search : Given a search query, the system finds the top 5 books that are most relevant based on their descriptions.
-  TF-IDF and LSI Models : Uses trained TF-IDF and LSI models to analyze book descriptions and generate similarity scores.
-  Book Descriptions : Provides a truncated description of each recommended book (first three sentences).
-  Relevance Score : Displays the relevance of each recommended book as a percentage based on the user's query.


## Setup Instructions

### Prerequisites
-  Python 3.7+ 
- Flask
- Pandas
- Spacy
- Gensim

### Installing Dependencies

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/book-recommendation-system.git
    cd book-recommendation-system
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the Spacy English language model:

    ```bash
    python -m spacy download en_core_web_sm
    ```

### Running the Application

1. Place the dataset `Book_Dataset_1.csv` in the project root directory.
2. If models (`models.pickle`) are not available, they will be trained automatically when the app is first run.
3. Start the Flask application:

    ```bash
    python app.py
    ```

4. Open your browser and navigate to `http://127.0.0.1:5000/` to access the application.

### Using the Application

1. On the home page, enter a search query related to a book (e.g., "mystery novel").
2. The app will return a list of books that are most relevant to your query, showing the book titles, truncated descriptions, and relevance scores.
3. Click on the images or links to learn more about each book.

### Dataset

The dataset used is `Book_Dataset_1.csv`, which contains the following columns:
-  Title : The title of the book.
-  Book_Description : The description or plot summary of the book.
-  Image_Link : A URL to the book cover image.

### Customization

-  Search Model : The system uses TF-IDF and LSI models for search queries. You can modify the `num_topics` parameter in the LSI model to adjust the granularity of topics.

### Example Queries

-  "Science fiction" 
-  "Romantic novels" 
-  "Historical mystery" 

