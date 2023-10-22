from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from predictions import client


# Create your views here.

class GetTraits(APIView):
    def post(self,request):
        data = request.data
        input_string = data.get('user_data')
        index = data.get('index')
        traits = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
        models = {}
        encoders = {}
        for trait in traits:
            model = load_model(f'/home/dev/Desktop/FYP/personality_model_{trait}.h5')
            with open(f'/home/dev/Desktop/FYP/label_encoder_{trait}.pkl', 'rb') as le_file:
                label_encoder = pickle.load(le_file)
            models[trait] = model
            encoders[trait] = label_encoder

        text_input = [f"{input_string}"]


        # Tokenize and pad the input text
        tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')  # Assuming max_words=1000
        tokenizer.fit_on_texts(text_input)
        input_sequence = tokenizer.texts_to_sequences(text_input)
        input_sequence = pad_sequences(input_sequence, maxlen=100)  # Assuming maxlen=100

        # Make predictions for each trait
        predictions = {}

        for trait in traits:
            model = models[trait]
            label_encoder = encoders[trait]
            
            predicted_prob = model.predict(input_sequence)[0][0]
            predicted_category = 'Yes' if predicted_prob >= 0.5 else 'No'
            predicted_score = predicted_prob * 100  # Assuming the score ranges from 0 to 100

            predictions[trait] = {
                'Probability': predicted_prob,
                'Category': predicted_category,
                'Score': predicted_score
            }

        response = []

        for trait, result in predictions.items():
            d = {
                "trait": trait,
                "probability": f"{result['Probability']:.2f}",
                "category":result["Category"],
                "score": f"{result['Score']:.2f}"
            }
            response.append(d)
        return Response({'response':response, 'index': index})
    


class FetchTweets(APIView):
    def post(self,request):
        data = request.data
        profile_url = data.get('profile_url')
        tweet_count = data.get('tweet_count')
        run_input = {
            "handles": [f"{profile_url}"],
            "tweetsDesired": tweet_count,
            "addUserInfo": True,
            "startUrls": [],
            "proxyConfig": { "useApifyProxy": True },
        }

        run = client.actor("quacker/twitter-scraper").call(run_input=run_input)

        response = []

        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            response.append(item)

        return Response(response)
