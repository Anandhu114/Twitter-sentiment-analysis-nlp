# Airline Twitter Sentiment Analysis Using Machine Learning

## 📌 Project Overview

Airline Twitter Sentiment Analysis is a Natural Language Processing (NLP) and Machine Learning project that analyzes tweets about airlines and classifies their sentiment as **positive, negative, or neutral**.

The project uses the airline sentiment dataset (`Tweets.csv`) and applies text preprocessing, feature extraction, and Logistic Regression to predict the sentiment expressed in each tweet.

It also includes exploratory data analysis (EDA), sentiment distribution charts, airline-wise sentiment comparisons, and word clouds to understand customer opinions and identify common words associated with positive and negative tweets.

## 🎯 Project Objectives

- Analyze customer feedback about airlines from Twitter data.
- Clean and preprocess tweet text for NLP.
- Explore tweet length, hashtags, URLs, and mentions.
- Understand Count Vectorization and TF-IDF feature extraction.
- Train a Logistic Regression classification model.
- Classify tweets as positive, negative, or neutral.
- Evaluate the model using accuracy, a classification report, and a confusion matrix.
- Visualize sentiment distributions and airline-wise sentiment percentages.
- Identify frequently used words in positive and negative tweets.

## 🛠️ Technologies and Libraries

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Regular Expressions (`re`) | Text pattern processing |
| NLTK | Tokenization and stopword removal |
| Scikit-learn | Feature extraction, model training, and evaluation |
| Logistic Regression | Sentiment classification |
| Matplotlib | Data visualization |
| Seaborn | Statistical charts and confusion matrix |
| Plotly | Interactive visualizations |
| WordCloud | Word cloud generation |
| Collections | Counting frequently occurring words |
| Jupyter Notebook | Interactive development environment |

## 📂 Dataset Description

The project uses `Tweets.csv`, which contains tweets about different airlines and their sentiment labels.

The notebook accesses the following columns:

| Column | Description |
|---|---|
| `tweet_id` | Identifier for a tweet |
| `text` | Original tweet text |
| `airline_sentiment` | Sentiment label: positive, negative, or neutral |
| `airline` | Airline associated with the tweet |

The notebook creates a structured DataFrame containing the tweet ID, original text, sentiment label, text length, and indicators for URLs, mentions, and hashtags.

## 🔄 Project Workflow

### 1. Importing Libraries

The notebook imports the Python libraries required for data processing, NLP, feature extraction, machine learning, evaluation, and visualization.

### 2. Loading the Dataset

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("Tweets.csv")

df.head()
```

The first few rows are inspected to understand the dataset structure and available columns.

### 3. Creating a Structured Dataset

The notebook creates a new DataFrame containing selected tweet information.

```python
structured_df = pd.DataFrame()

structured_df["id"] = df["tweet_id"]
structured_df["raw_text"] = df["text"]
structured_df["label"] = df["airline_sentiment"]

structured_df["length"] = df["text"].apply(len)
```

It also creates indicators for whether a tweet contains a URL, mention, or hashtag.

These additional features help explore the characteristics of the tweets.

### 4. Text Preprocessing

Text preprocessing prepares raw tweets for machine learning.

The notebook converts text to lowercase, tokenizes it using NLTK, and removes English stopwords.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def clean_and_tokenize(text):
    text = re.sub(r"^a-z\s", "", text.lower())
    tokens = word_tokenize(text)

    return " ".join(
        [word for word in tokens if word not in stop_words]
    )
```

The cleaned text is stored in a new `cleaned_text` column.

**Implementation note:** The regular expression in the current notebook does not remove all punctuation and special characters as intended. It can be improved before publishing the final project.

### 5. Text Feature Extraction

Machine learning algorithms require numerical input. The project explores ways to convert text into numerical features.

**Count Vectorization**

The notebook uses `CountVectorizer` to create binary word-presence features and word-count features.

**TF-IDF**

The notebook also creates a feature matrix using a variable named `tfidf_vec`. However, the current code initializes it with `CountVectorizer`, not `TfidfVectorizer`. Therefore, that section demonstrates count-based features rather than actual TF-IDF weighting.

For the main sentiment classification model, the notebook correctly uses `TfidfVectorizer`.

### 6. Exploratory Data Analysis (EDA)

The notebook visualizes the dataset using several charts:

- Histogram showing tweet-length distribution.
- Pie chart showing the percentage of tweets containing hashtags.
- Interactive pie chart showing the overall sentiment distribution.
- Interactive bar chart comparing sentiment percentages across airlines.
- Word clouds showing frequently used words in negative and positive tweets.
- Interactive word-frequency visualizations.

These visualizations help explore customer feedback and compare sentiment patterns.

### 7. Splitting the Dataset

The cleaned text is assigned to `X`, while the sentiment labels are assigned to `y`.

```python
from sklearn.model_selection import train_test_split

X = structured_df["cleaned_text"]
y = structured_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is divided into training and testing subsets. The test subset contains 20% of the data.

### 8. TF-IDF Vectorization

The project converts text into numerical features using TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
```

The vectorizer learns its vocabulary and feature weights from the training text. The same fitted vectorizer is then applied to the test text.

This helps avoid learning vocabulary information from the test set during feature extraction.

### 9. Training the Machine Learning Model

The project uses Logistic Regression for sentiment classification.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)
```

The model learns relationships between the TF-IDF features and the sentiment labels.

After training, it predicts the sentiment of the test tweets.

```python
y_pred = model.predict(X_test_vec)
```

### 10. Model Evaluation

The notebook evaluates the model using accuracy, a classification report, and a confusion matrix.

**Accuracy**

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")
```

Accuracy measures the proportion of test predictions that match the true labels.

**Classification Report**

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

The report includes precision, recall, F1-score, and support for each sentiment class.

**Confusion Matrix**

The notebook uses Seaborn to visualize the confusion matrix, comparing actual sentiment labels with predicted sentiment labels.

This helps identify which sentiment classes the model classifies correctly and which classes it confuses.

The notebook does not establish a final performance claim in this README; use the actual output from your completed run when reporting accuracy.

## ✨ Key Features

- Airline tweet data analysis.
- Text cleaning and tokenization.
- Stopword removal.
- Count-based text feature extraction.
- TF-IDF vectorization for classification.
- Logistic Regression sentiment classification.
- Training and testing split.
- Accuracy and classification report.
- Confusion matrix visualization.
- Overall sentiment distribution.
- Airline-wise sentiment analysis.
- Positive and negative word clouds.
- Interactive Plotly visualizations.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version.
- Jupyter Notebook or JupyterLab.
- The `Tweets.csv` dataset.

### 1. Clone the Repository

Replace the example URL with the URL of your actual GitHub repository.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy seaborn matplotlib plotly nltk scikit-learn wordcloud jupyter
```

### 4. Download the NLTK Resources

Run the following code in a notebook cell:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
```

Depending on your NLTK version, tokenization may also require:

```python
nltk.download("punkt_tab")
```

### 5. Add the Dataset

Place `Tweets.csv` in the same directory as the notebook, or update the file path in the code.

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `sentiment_analysis.ipynb` and execute the cells in order.

## 📁 Suggested Repository Structure

```text
airline-twitter-sentiment-analysis/
│
├── sentiment_analysis.ipynb
├── Tweets.csv
├── images/
│   ├── sentiment-distribution.png
│   ├── confusion-matrix.png
│   ├── airline-sentiment-comparison.png
│   └── sentiment-wordclouds.png
│
├── requirements.txt
└── README.md
```

This is a suggested structure, not a verified listing of the files currently in your repository. Check dataset licensing and file size before uploading `Tweets.csv` to GitHub.

## 💡 Applications

Sentiment analysis can help organizations:

- Understand customer feedback.
- Identify positive and negative customer experiences.
- Compare customer sentiment across airlines.
- Monitor recurring complaints and praise.
- Support customer experience analysis.
- Explore trends in customer opinions.

These are potential applications of sentiment analysis; the current notebook is an offline analysis and classification project, not a live Twitter monitoring system.

## 🔮 Future Improvements

- Correct the regular expression used for text cleaning.
- Use `TfidfVectorizer` in the section intended to demonstrate TF-IDF.
- Compare Logistic Regression with Naive Bayes, Support Vector Machines, and other classifiers.
- Use stratified splitting to preserve class proportions in the training and testing sets.
- Add cross-validation and hyperparameter tuning.
- Evaluate class-wise performance and investigate misclassified tweets.
- Save the fitted vectorizer and model for future predictions.
- Build a simple user interface using Streamlit or Flask.
- Add a function that accepts a new tweet and returns its predicted sentiment.
- Add verified evaluation results and screenshots to the README.

## 👨‍💻 Author

**Anandhu P S**

- GitHub: https://github.com/Anandhu114

## 📚 Key Concepts Demonstrated

- Natural Language Processing (NLP)
- Text Preprocessing
- Tokenization
- Stopword Removal
- Bag of Words
- TF-IDF
- Logistic Regression
- Multiclass Classification
- Model Evaluation
- Confusion Matrix
- Exploratory Data Analysis
- Sentiment Visualization

## 📄 License

No license is specified in the notebook. Add an appropriate license if you want other people to reuse, modify, or distribute your project.

---

If you find this project useful, feel free to star the repository!
