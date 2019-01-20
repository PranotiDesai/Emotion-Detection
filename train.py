import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from EmotionClassifer import EmotionClassifier
root_dir = "/users/imishra/workspace/EmotionDetection"
# Read the data
data_raw = pd.read_csv(root_dir+'/data/isear.csv', error_bad_lines=False, sep="|")
data = pd.DataFrame({'content': data_raw['SIT'], 'sentiment': data_raw['Field1']})

# Clean and transform the data
max_num_words = 4000
max_text_length = 1000
embed_dim = 128
lstm_units = 128
emotionClassifier = EmotionClassifier()
data['content'] = data['content'].apply(EmotionClassifier.clean_text)
data['sentiment_label'] = [emotionClassifier.emotions_labels_map[sentiment] for sentiment in data['sentiment']]

# Create and train the model
emotionClassifier.create_tokenizer(data['content'], max_num_words, max_text_length)
feature_vectors = emotionClassifier.map_features(data['content'])
labels = np.array(data['sentiment_label']).reshape(-1, 1)
emotionClassifier.fit_label_encoder(labels)
labels = emotionClassifier.encode_labels(labels)
emotionClassifier.create_model(embed_dim, lstm_units)
emotionClassifier.compile_model(loss_function='categorical_crossentropy', optimizer='rmsprop', metrics='accuracy')
X_train, X_valid, Y_train, Y_valid = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)
emotionClassifier.train(X_train, Y_train, X_valid, Y_valid, batch_size=128, epochs=30, verbose=2)

# Check performance on train data
pred = np.argmax(emotionClassifier.model.predict(X_train), axis=1)
Y = np.argmax(Y_train, axis=1)
confusionMatrix = pd.DataFrame(confusion_matrix(pred, Y))
confusionMatrix.columns = emotionClassifier.emotions_labels_map.keys()
confusionMatrix.index = emotionClassifier.emotions_labels_map.keys()
print("\n")
print(confusionMatrix)

# Check performance on test data
pred = np.argmax(emotionClassifier.model.predict(X_valid), axis=1)
Y = np.argmax(Y_valid, axis=1)
confusionMatrix = pd.DataFrame(confusion_matrix(pred, Y))
confusionMatrix.columns = emotionClassifier.emotions_labels_map.keys()
confusionMatrix.index = emotionClassifier.emotions_labels_map.keys()
print("\n")
print(confusionMatrix)



emotionClassifier.save_model(root_dir+'/model/lstm_model.h5')
emotionClassifier.model = None
output_file = root_dir+'/model/emotion_classifer.pkl'
with open(output_file, 'wb') as output:
    pickle.dump(emotionClassifier, output, pickle.HIGHEST_PROTOCOL)
