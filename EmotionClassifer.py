import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.preprocessing import OneHotEncoder


class EmotionClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None

        # most used n words
        self.max_num_words = None
        # max length of each text
        self.max_text_length = None
        # feature size of each vector
        self.feature_vector_size = None

        self.emotions_labels_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'guilt': 3, 'joy': 4, 'sadness': 5, 'shame': 6}
        self.labels_emotions_map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'guilt', 4: 'joy', 5: 'sadness', 6: 'shame'}

    @staticmethod
    def clean_text(text):
        """Clean_text"""
        text = re.sub('[^a-zA-z0-9\s]', '', text)
        #stop_words_list = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        #words = [re.sub("\\\\", '', word.lower()) for word in words if
        #         word.lower() not in stop_words_list and word.isalpha()]  # filter(lambda word: word not in stop_words_list, text.split())
        words = [re.sub("\\\\", '', word.lower()) for word in words if word.isalpha()]
        words = [lemmatizer.lemmatize(word) for word in words]

        text = " ".join(words)
        return text

    def fit_label_encoder(self, labels):
        """Fit the label enocder to one-hot encode the labels"""
        self.label_encoder = OneHotEncoder()
        self.label_encoder.fit(labels)

    def encode_labels(self, labels):
        """Encode the labels"""
        return self.label_encoder.transform(labels).toarray()

    def create_tokenizer(self, texts, max_num_words, max_text_length=500):
        """Fit a text tokenizer"""
        self.max_text_length = max_text_length
        self.max_num_words = max_num_words
        self.tokenizer = Tokenizer(num_words=max_num_words, lower=True, split=' ')
        self.tokenizer.fit_on_texts(np.array(texts, dtype=object))

    def map_features(self, texts):
        """Map text to feature vetcors"""
        feature_vectors = self.tokenizer.texts_to_sequences(np.array(texts, dtype=object))
        feature_vectors = pad_sequences(feature_vectors, maxlen=self.max_text_length)
        self.feature_vector_size = feature_vectors.shape[1];
        return feature_vectors

    def create_model(self, embed_dim, lstm_units):
        """Create a LSTM Neural Netwrok with Convolution layer"""
        self.model = Sequential()
        self.model.add(Embedding(self.max_num_words, embed_dim, input_length=self.feature_vector_size))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(SpatialDropout1D(0.5))
        self.model.add(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

    def compile_model(self, loss_function, optimizer, metrics):
        """Compile the model"""
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=[metrics])

    def train(self, X_train, Y_train, X_valid, Y_valid, epochs=10, batch_size=128, verbose=0):
        """Train the model"""
        self.model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=batch_size,
                       epochs=epochs, verbose=verbose)

    def classify(self, text):
        """Classify Emotions"""
        text = self.clean_text(text)
        feature_vectors = self.map_features([text])
        predicted_label = np.argmax(self.model.predict(feature_vectors), axis=1)
        return self.labels_emotions_map[predicted_label[0]]

    def load_model(self, model_path):
        """Load pre-trained model"""
        self.model = load_model(model_path)

    def save_model(self, model_path):
        self.model.save(model_path)
