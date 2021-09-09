# %%
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from google.cloud import storage


client = storage.Client()
# https://console.cloud.google.com/storage/browser/[bucket-id]/
bucket = client.get_bucket('moritz-bucket')
# Then do other things...
blob = bucket.get_blob('cleanbook.txt') #.decode("utf-8")
lines = blob.download_as_text().split("\n")

# print(lines)
# print(type(lines))

# %%

words = "".join(lines).split()
words = [x.lower() for x in words]

tups = []
for idx, word in enumerate(words[:-1]):
    tups.append(
        (word, words[idx+1])
    )


#%%
sentences = []
for l in lines:
    sentences.extend(l.split("."))

#%%
tokenizer = tf.keras.preprocessing.text.Tokenizer()


def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


inp_sequences, total_words = get_sequence_of_tokens(sentences)


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = tf.keras.utils.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)




# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 20, input_length=max_sequence_len),
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=total_words, activation='softmax'),
])

model.compile('adam', 'categorical_crossentropy')

model.fit(predictors, label, epochs=3, batch_size=64)

# %%
