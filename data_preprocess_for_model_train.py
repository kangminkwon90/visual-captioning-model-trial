"""
data preparation for model train
    image data: load, resize, to array
    text data: regularize, tokenize, to sequences, padding, int encoding(label)
    train_test_split
"""

"""
(module import example)

from data_preprocess_for_model_train import data_for_model_train

df = pd.read_csv('./train_label/train_sample_2000.csv')
image_folder = './train_image'

(train_image, test_image,
train_sequences, test_sequences,
train_label, test_label,
 vocab_size, tokenizer, max_sequence_length) = data_for_model_train(df, image_folder, test_size=0.05)
"""

# import list
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
import re
import os

# text data preprocessing func

def preprocess_and_tokenize(df, text_column='sentence_en'):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", "", text)
        return text

    start_token = '<start>'
    end_token = '<end>'
    pad_token = '<pad>'
    unk_token = '<unk>'
    df['cleaned_sentence'] = df[text_column].apply(clean_text)

    tokenizer = Tokenizer(oov_token=unk_token)
    tokenizer.fit_on_texts(df['cleaned_sentence'])

    tokenizer.word_index[start_token] = len(tokenizer.word_index) + 1
    tokenizer.word_index[end_token] = len(tokenizer.word_index) + 1
    tokenizer.word_index[pad_token] = 0
    tokenizer.word_index[unk_token] = len(tokenizer.word_index) + 1
    tokenizer.index_word[0] = pad_token
    tokenizer.index_word[len(tokenizer.word_index)] = unk_token

    sequences = tokenizer.texts_to_sequences(df['cleaned_sentence'])
    sequences = [[tokenizer.word_index[start_token]] + seq + [tokenizer.word_index[end_token]] for seq in sequences]

    max_length = max(len(sequences) for sequence in sequences)
    sequences_padded = pad_sequences(sequences,
                                     maxlen=max_length,
                                     padding='post',
                                     value=tokenizer.word_index[pad_token])

    vocab_size = len(tokenizer.word_index) + 1

    x_data_text = sequences_padded[:, :-1]
    y_data_label = sequences_padded[:, 1:]
    max_sequence_length = max_length - 1

    return x_data_text, y_data_label, vocab_size, tokenizer, max_sequence_length

# image data preprocessing func

def preprocess_image(image_folder, file_name, target_size=(299,299)):
    image_path = os.path.join(image_folder, file_name)
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    return image

# train_test_split func

def data_for_model_train(df, image_folder, test_size):
    image_folder = image_folder
    (x_data_text, y_data_label,
     vocab_size, tokenizer,
     max_sequence_length) = preprocess_and_tokenize(df)
    x_data_image = np.array([preprocess_image(image_folder, file_name) for file_name in df['file_name']])

    (train_image, test_image,
     train_sequences, test_sequences,
     train_label, test_label) = train_test_split(x_data_image, x_data_text, y_data_label,
                                                 test_size=test_size, random_state=42)

    print(train_image.shape)
    print(train_sequences.shape)
    print(train_label.shape)
    print(test_image.shape)
    print(test_sequences.shape)
    print(test_label.shape)

    return (train_image, test_image,
            train_sequences, test_sequences,
            train_label, test_label,
            vocab_size, tokenizer, max_sequence_length)

#### test module

# df = pd.read_csv('./train_label/train_sample_2000.csv')
# image_folder = './train_image'
#
# (train_image, test_image,
# train_sequences, test_sequences,
# train_label, test_label,
#  vocab_size, tokenizer, max_sequence_length) = data_for_model_train(df, image_folder, test_size=0.05)