# ==========================
# TEXT GENERATION USING LSTM
# ==========================

# 1. IMPORT LIBRARIES
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import string

# 2. LOAD DATASET

print("Loading dataset...")

file = open("content.txt", "r", encoding="utf-8")
text = file.read()
file.close()

print("Dataset loaded successfully!")

# 3. PREPROCESSING

print("Preprocessing text...")

text = text.lower()

translator = str.maketrans('', '', string.punctuation)
text = text.translate(translator)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Total unique words:", total_words)

# 4. CREATE INPUT SEQUENCES

print("Creating input sequences...")

input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        ngram = token_list[:i+1]
        input_sequences.append(ngram)

print("Total sequences created:", len(input_sequences))

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# 5. TRAIN TEST SPLIT

print("Splitting dataset...")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 6. BUILD MODEL

print("Building model...")

model = Sequential()

model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# 7. TRAIN MODEL

print("Training started...")

early_stop = EarlyStopping(monitor='val_loss', patience=3)

checkpoint = ModelCheckpoint(
    "best_text_generator.h5",
    monitor='val_loss',
    save_best_only=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop, checkpoint]
)

print("Training completed!")

# 8. TEXT GENERATION FUNCTION

def generate_text(seed_text, next_words):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


# 9. TEST THE MODEL

print("\n===== GENERATED TEXT =====\n")

print(generate_text("to be or not to", 20))
print(generate_text("romeo and juliet", 20))
print(generate_text("the king said", 20))
