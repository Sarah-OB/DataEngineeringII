import unittest
import os
import requests
from bs4 import BeautifulSoup


class FlaskTests(unittest.TestCase):
    def setUp(self):
        os.environ['NO_PROXY'] = '0.0.0.0'
        self.sentiment_positive = "Positive"
        self.sentiment_negative = "Negative"
        self.sentiment_neutral = "Neutral"

        self.emoji_positive = "😁"
        self.emoji_negative = "🙁"
        self.emoji_neutral = "😐"
        pass

    def tearDown(self):
        pass

    def test_index(self):
        responce = requests.get('http://localhost:5000')
        self.assertEqual(responce.status_code, 200)

    def test_predict_page(self):
        responce = requests.get('http://localhost:5000')
        self.assertEqual(responce.status_code, 200)

    def test_get_sentiment_positive(self):
        params = {
            'message': 'I am happy',
            'message_user': "analyze_message",
        }

        expected = '<h3 class="text-center">The sentiment of your text is ' + self.sentiment_positive + ' <p style="font-size:100px">' + self.emoji_positive +'</p></h3>'
        responce = requests.post('http://localhost:5000/predict', data=params)
        self.assertEqual(responce.status_code, 200)
        result = requests.get('http://localhost:5000/predict').text
        reponse_positive = BeautifulSoup(responce.content, 'html.parser')
        positive_rep = reponse_positive.find(class_="text-center")
        self.assertEqual(str(positive_rep), expected)

    def test_get_sentiment_negative(self):
        params = {
            'message': 'I hate cats',
            'message_user': "analyze_message",
        }

        expected = '<h3 class="text-center">The sentiment of your text is ' + self.sentiment_negative + ' <p style="font-size:100px">' + self.emoji_negative +'</p></h3>'
        responce = requests.post('http://localhost:5000/predict', data=params)
        self.assertEqual(responce.status_code, 200)
        result = requests.get('http://localhost:5000/predict').text
        reponse_negative = BeautifulSoup(responce.content, 'html.parser')
        negative_rep = reponse_negative.find(class_="text-center")
        self.assertEqual(str(negative_rep), expected)

    def test_get_sentiment_neutral(self):
        params = {
            'message': 'Trump lose the election',
            'message_user': "analyze_message",
        }

        expected = '<h3 class="text-center">The sentiment of your text is ' + self.sentiment_neutral + ' <p style="font-size:100px">' + self.emoji_neutral +'</p></h3>'
        responce = requests.post('http://localhost:5000/predict', data=params)
        self.assertEqual(responce.status_code, 200)
        result = requests.get('http://localhost:5000/predict').text
        reponse_neutral = BeautifulSoup(responce.content, 'html.parser')
        neutral_rep = reponse_neutral.find(class_="text-center")
        self.assertEqual(str(neutral_rep), expected)


if __name__ == '__main__':
    unittest.main()
