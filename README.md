# News Content NLP Projects - Fake News Detection and Topic Modeling

This is a news content NLP analysis written by Wally, Huang Lin, Chun. The project structure includes two main sections, one is the **Fake News Detection by Supervised ML based on Tfidf**, and the other is the **Sentiment Analysis with nltk and Topic Modeling with Gensim**.

## Fake News Detection by Supervised ML based on Tfidf

In the first section, we aim at predicting fake news based on the text structures and vectors. We first use the `TfidfVectorizer()` <span style="color:skyblue">(Term Frequency - Inverse Document Frequency) </span>method built in the `sklearn` machine learning tool kit to extract features from raw texts.

Exploiting the ***tfidf*** of each document as feature matrix, we trained three candidate traditional machine learning models and selected the best-performing one as the final model to optimize.

Three models selected as candidates are:

1. Logistic Regression ($\because$ Efficiency)
2. Random Forest Classifier ($\because$ Versatility)
3. Multinomial NB ($\because$ Strength of text classification)

![plot](./1_ipynb/2_cross_validation.png)