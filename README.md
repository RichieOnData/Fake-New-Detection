## Detecting Fake News: A Machine Learning Approach

This project tackles the growing threat of misinformation by leveraging machine learning models to classify news as real or fake. Using the ISOT Fake News Dataset, it explores a variety of algorithms including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost — with XGBoost emerging as the top performer. The solution features TF-IDF vectorization, Intel-optimized code, and thorough model evaluation using precision, recall, F1-score, and ROC-AUC. Future work will include testing additional models and vectorizers, optimizing hyperparameters, and expanding to diverse datasets.

- The ISOT Fake News dataset can be downloaded from this link:
https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

  

## Deliverables
We have addressed the following:
- Data Collection: The ISOT Fake News Dataset was used. Exploratory Data Analysis was performed on the dataset.
- Data Preparation: The data was cleaned and prepared for training.
- Model Training: Various machine learning models were trained on the labeled news data.
- Model Evaluation: The performance of the models was assessed based on some important metrics.
- Parameter Tuning: Model parameters were tweaked to improve performance.
- Model Selection: The best-performing model was selected based on evaluation metrics.
- Making Predictions: The best model was used to classify new articles as real or fake.

## Models
Several machine learning models can be used in a fake news detection project, but the models we chose are:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Passive Aggressive Classifier
- XG Boost

Parameter tuning and patching were done on the logistic regression model and it showed that Intel optimization did in fact help to improve our code runtime.

## Evaluation

The performance of the trained models was evaluated using various metrics such as accuracy, precision, recall, F1 score, AUC ROC score. Confusion matrices were also plotted to visualize the performance of the models.

