import random
import re
import string
import pickle
import pandas as pd
import nltk

from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load dataset
ne = pd.read_csv('TweetsCSV/processedNegative.csv')
po = pd.read_csv('TweetsCSV/processedPositive.csv')
na = pd.read_csv('TweetsCSV/processedNeutral.csv')

# Preprocessing
N = []
for i in ne.columns:
    N.append(i)

P = []
for i in po.columns:
    P.append(i)

NA = []
for i in na.columns:
    NA.append(i)

# Tokenization
positive_tweet_tokens = [word_tokenize(i) for i in P]
negative_tweet_tokens = [word_tokenize(i) for i in N]
neutral_tweet_tokens = [word_tokenize(i) for i in NA]


# Lemmatization
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


# Stop words
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


stop_words = stopwords.words('english')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []
neutral_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in neutral_tweet_tokens:
    neutral_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
neutral_tokens_for_model = get_tweets_for_model(neutral_cleaned_tokens_list)

# Dataset construction
positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

neutral_dataset = [(tweet_dict, "Neutral")
                   for tweet_dict in neutral_tokens_for_model]

dataset = positive_dataset + negative_dataset + neutral_dataset

random.shuffle(dataset)

# Train test
random.shuffle(dataset)
train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

# Trainning the model : NLTK NaiveBayes
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))

# Testing
custom_tweet = "I love me "

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))

#Serialize the model
pickle.dump(classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))
