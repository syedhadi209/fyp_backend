import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tensorflow.keras.models import load_model


data = pd.read_csv('data/mypersonality_final.csv',encoding="ISO-8859-1")


# Encode categorical columns
label_encoders = {}
categorical_columns = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])


# Text Data Processing
max_words = 1000  # Adjust as needed
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['STATUS'])
sequences = tokenizer.texts_to_sequences(data['STATUS'])
X = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed


# Create dictionaries to store models and encoders
models = {}
encoders = {}

# Iterate through each trait
for trait in categorical_columns:
    print(f"Training model for {trait}")
    
    # Model Building
    model = Sequential()
    model.add(Embedding(input_dim=max_words + 1, output_dim=100, input_length=X.shape[1]))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification for each trait

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, data[trait], test_size=0.2, random_state=42)

    # Model Training
    batch_size = 32
    epochs = 25
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=2)

    # Save the model and encoder
    models[trait] = model
    encoders[trait] = label_encoders[trait]

    # Inference
    text_input = ["Your input text here"]  # Replace with your text input
    input_sequence = tokenizer.texts_to_sequences(text_input)
    input_sequence = pad_sequences(input_sequence, maxlen=100)
    predicted_prob = model.predict(input_sequence)[0][0]
    predicted_category = 'Yes' if predicted_prob >= 0.5 else 'No'
    predicted_score = predicted_prob * 100  # Assuming the score ranges from 0 to 100

    print(f'Predicted Probability for {trait}: {predicted_prob:.2f}')
    print(f'Predicted Category for {trait}: {predicted_category}')
    print(f'Predicted Score for {trait}: {predicted_score:.2f}')



# Save the models and encoders to pickle files
for trait in categorical_columns:
    model_filename = f'personality_model_{trait}.h5'
    encoder_filename = f'label_encoder_{trait}.pkl'
    models[trait].save(model_filename)
    with open(encoder_filename, 'wb') as le_file:
        pickle.dump(encoders[trait], le_file)