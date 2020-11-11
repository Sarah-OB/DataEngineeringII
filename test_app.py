import unittest
import os
import requests
import BeautifulSoup


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

    def test_get_sentiment_positive(self):
        params = {
            'message': 'I am happy',
            'message_user': "analyze_message",
        }

        responce = requests.post('http://localhost:5000/predict', data=params)
        self.assertEqual(responce.status_code, 200)
        result = requests.get('http://localhost:5000/result')
        reponse_page = BeautifulSoup(result.data, 'html.parser')
        positive_rep = reponse_page.find(id_='reponse')
        self.assertEqual(str(positive_rep), '<span class="text-center" id="reponse">' + self.sentiment_positive.encode() + '</span>')

    def test_get_sentiment_negative(self):
        params = {
            'message': 'I hate cats',
            'message_user': "analyze_message",
        }

        responce = requests.post('http://localhost:5000/predict', data=params)
        self.assertEqual(responce.status_code, 200)
        self.assertEqual(responce.content, self.sentiment_negative.encode())

    def test_get_sentiment_neutral(self):
        params = {
            'message': 'I am walking on the street',
            'message_user': "analyze_message",
        }

        responce = requests.post('http://localhost:5000/predict', data=params)
        self.assertEqual(responce.status_code, 200)
        self.assertEqual(responce.content, self.sentiment_neutral.encode())


if __name__ == '__main__':
    unittest.main()
