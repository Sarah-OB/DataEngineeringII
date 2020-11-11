from flask import Flask, request, render_template
import pickle
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


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


def get_sentiment(message):
    status = "fail"
    responce = ""
    emoji = ""
    message = remove_noise(word_tokenize(message))
    responce = model.classify(dict([token, True] for token in message))

    if responce is not '':
        status = "success"

        if responce == 'Positive':
            emoji = "😁"
        elif responce == 'Negative':
            emoji = "🙁"
        else:
            emoji = "😐"

    return status, responce, emoji


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    status = 'fail'
    prediction = ''

    if request.method == 'POST':
        text = request.form
        if text['message_user'] == 'analyze_message' and text['message'] is not '':
            status, prediction, emojis = get_sentiment(text['message'])

            if status is 'success':
                return render_template('result.html',
                                       sentiment_responce="The sentiment of your text is {}".format(prediction), emoji = str(emojis))
            else:
                return render_template('index.html', error="We didn't succeed to analyze your text, please try again.")

        else:
            return render_template('index.html', error="We can't analyze empty text.")


@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
