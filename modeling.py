"""
define image captioning model
compile model

structure summary
    LSTM units: 256
    image features: InceptionResNetV2(pre-trained weights)
    text: Bidirectional LSTM, LSTM
    attention: cross attention, self-attention

params: vocab_size, lstm_units, max_sequence_length
"""

# import list
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import AdditiveAttention, MultiHeadAttention, Lambda, Reshape
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam

# modeling func

def create_model(vocab_size, lstm_units, max_sequence_length):
    inception_resnet_model = InceptionResNetV2(include_top=False,
                                               weights='imagenet',
                                               pooling='avg')
    inception_resnet_model.trainable = False

    image_input = Input(shape=(299, 299, 3))
    image_features = inception_resnet_model(image_input)

    sequence_input = Input(shape=(max_sequence_length,))
    embedding = Embedding(input_dim=vocab_size,
                          output_dim=256,
                          mask_zero=True)(sequence_input)

    bidirectional_lstm = Bidirectional(LSTM(lstm_units,
                                            return_sequences=True))
    lstm_output = bidirectional_lstm(embedding)

    dropout = Dropout(0.5)
    lstm_output = dropout(lstm_output)

    second_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    lstm_output, _, _ = second_lstm(lstm_output)

    image_features_dense = Dense(lstm_units)(image_features)
    image_features_dense = Reshape((1, lstm_units))(image_features_dense)
    image_features_dense = Lambda(lambda x: tf.tile(x, [1, max_sequence_length, 1]))(image_features_dense)

    print("LSTM output shape: ", lstm_output.shape)
    print("Image features shape: ", image_features_dense.shape)

    attention = AdditiveAttention()
    context_vector, attention_weights = attention([lstm_output, image_features_dense],
                                                  return_attention_scores=True)
    self_attention = MultiHeadAttention(num_heads=8, key_dim=lstm_units)
    self_attention_output = self_attention(query=lstm_output,
                                           value=lstm_output,
                                           key=lstm_output)
    combined_attention_output = tf.keras.layers.Concatenate(axis=-1)([context_vector, self_attention_output])

    dense = Dense(vocab_size, activation='softmax')
    output = dense(combined_attention_output)

    return Model(inputs=[image_input, sequence_input], outputs=output)

# create and compile model

def compile_model(model):
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

## test module

from data_preprocess_for_model_train import data_for_model_train
import pandas as pd
import numpy as np

df = pd.read_csv('./train_label/train_sample_2000.csv')
image_folder = './train_image'

(train_image, test_image,
train_sequences, test_sequences,
train_label, test_label,
 vocab_size, tokenizer, max_sequence_length) = data_for_model_train(df, image_folder, test_size=0.05)

model_initial = create_model(vocab_size, 256, max_sequence_length)
model = compile_model(model_initial)
initial_weights = model.get_weights()
model.summary()
