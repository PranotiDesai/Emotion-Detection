import pickle
from flask import Flask, jsonify, request
from EmotionClassifer import EmotionClassifier
app = Flask(__name__)
cache = {}  # To let the model persist


@app.route('/classify', methods=['POST'])
def classify():
    """
    This services calls the learning model classify method to classify the sentiment
    """
    try:
        # get teh request parameter
        json_data = request.get_json()
        data = json_data
        print(json_data)
        text = data['text']

    except Exception:
        return jsonify(result="error", emotion="")

    # If the parameters are good then load the model if not loaded
    if len(cache) == 0:
        with open('model/emotion_classifer.pkl', 'rb') as input:
            model = pickle.load(input)
            model.load_model('model/lstm_model.h5')

            #  Causes error without it in some version of keras
            model.tokenizer.oov_token = None

        cache["model"] = model

    emotion = cache["model"].classify(text)
    return jsonify(result="success", emotion=emotion)


if __name__ == '__main__':
    app.run(debug=True)
