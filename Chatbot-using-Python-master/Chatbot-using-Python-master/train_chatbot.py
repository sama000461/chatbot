import nltk
import os
import shutil
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


def fix_nltk():
    # Set the correct NLTK data path
    nltk_data_path = os.path.join(os.environ['APPDATA'], 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # Download required NLTK data
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)
    
    # Create punkt_tab symlink if needed
    punkt_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt')
    punkt_tab_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
    
    if not os.path.exists(punkt_tab_path) and os.path.exists(punkt_path):
        try:
            os.symlink(punkt_path, punkt_tab_path)
        except:
            # If symlink fails, copy the directory
            shutil.copytree(punkt_path, punkt_tab_path)

# Apply the fix before any NLTK processing
fix_nltk()

# ================== here is the main code ================== #
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load intents
try:
    data_file = open('intents.json').read()
    intents = json.loads(data_file)
except Exception as e:
    print(f"Error loading intents.json: {e}")
    exit()

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        try:
            # Tokenize each word (now should work after the fix)
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
        except Exception as e:
            print(f"Error processing pattern '{pattern}': {e}")
            continue

# Lemmatize and clean words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words")

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)  

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model created successfully")