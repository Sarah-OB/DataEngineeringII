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

        expected = '<span class="text-center" id="reponse">The sentiment of your text is ' + self.sentiment_positive + '</span>'
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

        expected = '<span class="text-center" id="reponse">The sentiment of your text is ' + self.sentiment_negative + '</span>'
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

        expected = '<span class="text-center" id="reponse">The sentiment of your text is ' + self.sentiment_neutral + '</span>'
        responce = requests.post('http://localhost:5000/predict', data=params)
        self.assertEqual(responce.status_code, 200)
        result = requests.get('http://localhost:5000/predict').text
        reponse_neutral = BeautifulSoup(responce.content, 'html.parser')
        neutral_rep = reponse_neutral.find(class_="text-center")
        self.assertEqual(str(neutral_rep), expected)


if __name__ == '__main__':
    unittest.main()
