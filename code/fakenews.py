import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import nltk
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from timeit import default_timer as timer
from sklearnex import patch_sklearn, unpatch_sklearn
from IPython.display import HTML

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

fake_data = pd.read_csv('C:/Users/John Peter/Downloads/intel-project/project/4BDA/data/News-_dataset/Fake.csv')
fake_data.head()

true_data = pd.read_csv('C:/Users/John Peter/Downloads/intel-project/project/4BDA/data/News-_dataset/True.csv')
true_data.head()

fake_data['label'] = 0
true_data['label'] = 1
data = pd.concat([true_data, fake_data], axis=0)

plt.figure(figsize = (6,4))
sns.set(style = "whitegrid",font_scale = 1.0)
chart = sns.countplot(x = "subject", data = data)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)

data['text'] = data['title'] +' '+data['text']
del data['title']
del data['subject']
del data['date']
data.head()
data.shape
data.isnull().sum() # get the count of missing/NULL values for each column. if present remove missing values
#Shuffling the data by sampling it randomly, then resetting the index and dropping the previous index column
data = data.sample(frac=1).reset_index(drop=True)
data.head()

sns.countplot(data=data,
              x='label',
              order=data['label'].value_counts().index)

#data cleaning
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize text into individual words
    tokens = word_tokenize(text)

    # Remove stopwords
    custom_stopwords = stopwords.words('english')
    tokens = [word for word in tokens if word not in custom_stopwords]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in tokens)

    return text

data['text'] = data['text'].apply(preprocess_text)

x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.3, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

def evaluate_model(model, x_train, y_train, x_test, y_test):
    # Train the model
    model.fit(x_train, y_train)

    # Predict on the training and test data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    def measure_error(y_true, y_pred, label, x_data):
        return pd.Series({
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'ROC AUC': roc_auc_score(y_true, y_pred)
        }, name=label)

    # Calculate evaluation metrics
    train_metrics = measure_error(y_train, y_train_pred, 'Train', x_train)
    test_metrics = measure_error(y_test, y_test_pred, 'Test', x_test)

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Plot confusion matrices
    cm_display_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=[False, True])
    cm_display_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[False, True])

    print("Training Data:")
    print(train_metrics)
    cm_display_train.plot()
    plt.show()

    print("Test Data:")
    print(test_metrics)
    cm_display_test.plot()
    plt.show()

    return model

patch_sklearn()
params = {
    'C': 0.1,
    'solver': 'lbfgs',
    'multi_class': 'multinomial',
    'n_jobs': -1,
}

start = timer()
model_patched = LogisticRegression(**params).fit(x_train, y_train)
train_patched = timer() - start
f"Intel® extension for Scikit-learn time: {train_patched:.2f} s"
evaluate_model(model_patched, x_train, y_train, x_test, y_test)