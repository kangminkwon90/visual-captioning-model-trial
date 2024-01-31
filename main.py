# main.py

"""
[Annotation: module structure]
1) Python version: 3.10.11
2) Frameworks: Tensorflow 2.15
3) modules used in main.py
    data_preprocess_for_model_train.py
    modeling.py
    train_validation.py

[Annotation: about model save)]

Because of storage shortage, train loop function does not save model for each train loop.
Instead, ModelTrainer class returns dictionaries of history, test & predicted data(sequences, text), and tokenizer.
Therefore, without loading specific model.h5 file, results would be validated.

[Annotation: about WARNING or ERROR logs for package import]

when importing tensorflow packages and using them with absl, there might be several warning, error messages.
1) NUMA warning: NUMA is a tool for optimizing memory allocation.
    It is normally used in massive server group, therefore, if you use personal device, just ignore it

2) Type Inference failed: It can be caused by type differences between X_data and Y_data.
    In this model, X_data(image features, text sequences) type is float32 and Y_data(text label) type is int32.
    For this difference, the system might be confused because it is ordered to predict int-type values but received float.
    This conflict can be solved by changing value type of Y_data before train(tf.cast(y, tf.float32)).
    However, when you use "sparse-categorical-crossentropy" as loss function, Y_data must be type int32.
    Therefore, if model train goes well, this error can be ignored.
    Otherwise, it is caused by other problems such as model structure and you must fix this.

3) In conclusion, if model training goes well, you can ignore them. They are just matter of system optimization.

4) If you want them not to be printed. Use this configuration code.
    <tensorflow>
        import os
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    <absl>
        from absl import logging
        logging.set_verbosity(logging.ERROR)


[Flow]
<data load>
    label csv
    image files

<data preparation>
    (initial vars)
    df(label csv)
    image_folder(image file path)

    (function)
    data_for_model_train(df, image_folder, test_size)

    (output vars)
    train_image, test_image, train_sequensces, test_sequences,
    train_label, test_label,
    vocab_size, tokenizer, max_sequence_length

<model create>
    (initial vars)
    vocab_size, max_sequence_length

    (function)
    create_model(vocab_size, lstm_units=256, max_sequence_length)
    compile_model(model)

    (output vars)
    model

<model train>
    (initial vars)
    declare list(train loop params): epochs_list, train_data_volume
    output vars(from earlier process):
        train_image, train_sequences, train_label,
        test_image, test_sequences, test_label, tokenizer

    (class, function)
    class: ModelTrainer(model, train_image, train_sequences, train_label,
                        test_image, test_sequences, test_label,
                        epochs_list, train_data_volume, tokenizer)
    function(method in ModelTrainer class): trainer

    (output vars)
    history_dict, predictions_dict, predictions_text_dict,
    test_label, bleu_score_dict, test_label_text

<save outputs>
    (initial vars)
    declare str: filename(_h5, json)
    output vars:
        predictions_dict
        history_dict
        test_label
        bleu_score_dict
        tokenizer
        predictions_text_dict
        test_label_text

    (function)
    save_predictions_to_hdf5(predictions_dict, filename_h5)
    save_history_dict_to_hdf5(history_dict, filename_h5)
    save_test_label_to_hdf5(test_label, filename_h5)
    save_bleu_to_hdf5(bleu_score_dict, filename_h5)
    save_converted_text_to_json(predictions_text_dict, test_label_text, pred_json, label_json)
    save_tokenizer_to_json(tokenizer, filename_json)

    (outputs: produced in same dir of 'main.py')
    predictions_dict.h5, history_dict.h5,
    test_label.h5, bleu_score.h5,
    tokenizer.json, predictions_text_dict.json, test_label_text.json

<model validation: visualize scores>
    (initial vars)
    history_dict, bleu_score_dict, train_data_volume
    (output vars)
    df_scores

    (function)
    scores_to_df(history_dict, bleu_score_dict, train_data_volume)
    scores_to_plot(df_scores, image_name_jpg)

    (outputs: produced in same dir of 'main.py')
    image_name.jpg(scores plot graph image file)

<model validation: visualize test image with captions(actual, predicted)>
    (initial vars)
    test_image, predictions_text_dict, test_label_text,
    index(nums of printing rows within test data size),
    key(select certain predicted texts in predicted_text_dict, format: (epochs, train_data_volume)

    (function)
    display_image_with_caption(test_image, predictions_text_dict, test_label_text, index, key)

    (outputs)
    printed str: predicted caption, actual caption
    printed img: test image
"""

# import list
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import time
import h5py
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt')
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import AdditiveAttention, MultiheadAttention, Lambda, Reshape
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam

from data_preprocess_for_model_train import data_for_model_train
from modeling import create_model, compile_model
import train_validation

# data load
    # (later) replace to git clone

df = pd.read_csv('./train_label/train_sample_2000.csv', encoding='utf-8')
image_folder = './train_image'

# preprocess and prepare data for model train

(train_image, test_image,
 train_sequences, test_sequences,
 train_label, test_label,
 vocab_size, tokenizer, max_sequence_length) = data_for_model_train(df, image_folder, test_size=0.05)

# create and compile model

model_initial = create_model(vocab_size=vocab_size, lstm_units=256, max_sequence_length=max_sequence_length)
model = compile_model(model_initial)

# train model with loop params

train_data_volume = [200, 800, 1000, 1400, 1600, 1900]
epochs_list = [20, 30, 40]

trainer = train_validation.ModelTrainer(model, train_image, train_sequences, train_label,
                                        test_image, test_sequences, test_label,
                                        epochs_list, train_data_volume, tokenizer)

(history_dict, predictions_dict, predictions_text_dict,
 test_label, bleu_score_dict, test_label_text) = trainer.train()


# save output results

train_validation.save_predictions_to_hdf5(predictions_dict, 'predictions_dict.h5')
train_validation.save_history_dict_to_hdf5(history_dict, 'history_dict.h5')
train_validation.save_test_label_to_hdf5(test_label, 'test_label.h5')
train_validation.save_bleu_to_hdf5(bleu_score_dict, 'bleu_score_dict.h5')
train_validation.save_tokenizer_to_json(tokenizer, 'tokenizer.json')
train_validation.save_converted_text_to_json(predictions_text_dict, test_label_text,
                                             'predicted_text.json',
                                             'test_label_text.json')

# visualize metrics scores result(with convert to dataframe)

df_scores = train_validation.scores_to_df(history_dict, bleu_score_dict, train_data_volume)
df_scores.to_csv('./df_scores.csv', encoding='utf-8', index=False)

train_validation.scores_to_plot(df_scores, 'metrics_bleu_plots.jpg')

# visualize validation(test images - predicted captions - actual captions)

key_for_valid = (40, 1900)
index = 10

train_validation.display_image_with_caption(test_image,
                                            predictions_text_dict,
                                            test_label_text,
                                            index=index,
                                            key=key_for_valid)