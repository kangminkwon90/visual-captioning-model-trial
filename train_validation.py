"""
model train loop class
    training model according to epochs list and data volume list
    label predictions include
    convert label and predicted sequences include
    calculate bleu score include

validation data print func
    print test image - test label - predicted caption

metrics scores visualize func
    plot graph: accuracy, loss, val_accuracy, val_loss
    plot graph: bleu score

save outputs func
    hdf5 format: history dict, predictions dict, test label, bleu score dict
    json format: predictions(text converted) dict, tokenizer
"""

# import list

import time
import tensorflow as tf
import pandas as pd
import numpy as np
import json
from keras.preprocessing.text import tokenizer_from_json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt')
import h5py
import matplotlib.pyplot as plt

# train loop class

class ModelTrainer:
    def __init__(self, model,
                 train_image, train_sequences, train_label,
                 test_image, test_sequences, test_label,
                 epochs_list, train_data_volume, tokenizer):

        self.model = model
        self.initial_weights = model.get_weights()
        self.train_image = train_image
        self.train_sequences = train_sequences
        self.train_label = train_label
        self.test_image = test_image
        self.test_sequences = test_sequences
        self.test_label = test_label
        self.epochs_list = epochs_list
        self.train_data_volume = train_data_volume
        self.tokenizer = tokenizer
        self.index_to_word = {index: word for word, index in self.tokenizer.word_index.items()}
        self.end_token_id = tokenizer.word_index.get('<end>')

        self.history_dict = {}
        self.predictions_dict = {}
        self.predictions_text_dict = {}
        self.bleu_score_dict = {}

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs):
            super().__init__()
            self.epochs = epochs
            self.start_time = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            end_time = time.time()
            epoch_duration = end_time = self.start_time
            if epoch == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1}/{self.epochs} - {epoch_duration: .2f} sec", logs)

    def convert_label_to_text(self, test_label):
        end_token_id = self.tokenizer.word_index['<end>']
        text_output = []
        for sequence in test_label:
            sequence_text = []
            for idx in sequence:
                if idx == end_token_id:
                    break
                if idx > 0:
                    sequence_text.append(self.index_to_word.get(idx, ''))
            text_output.append(' '.join(sequence_text))
        return text_output

    def convert_predictions_to_text(self, predicted_sequences):
        end_token_id = self.tokenizer.word_index['<end>']
        predicted_text_output = []

        for sequence in predicted_sequences:
            sequence_text = []
            for word_probabilities in sequence:
                idx = np.argmax(word_probabilities)
                if idx == end_token_id:
                    break
                word = self.tokenizer.index_word.get(idx, '')
                if word:
                    sequence_text.append(word)
            predicted_text_output.append(' '.join(sequence_text))
        return predicted_text_output

    def calculate_bleu(self, references, candidates):
        score = 0
        smoothie = SmoothingFunction().method4
        for ref, cand in zip(references, candidates):
            ref_tokens = nltk.word_tokenize(ref.lower())
            cand_tokens = nltk.word_tokenize(cand.lower())
            score += sentence_bleu([ref_tokens],
                                   cand_tokens,
                                   weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=smoothie)
        return round((score / len(candidates)), 3)

    def train(self):

        for epochs in self.epochs_list:
            history_list = []
            for data_volume in self.train_data_volume:
                self.model.set_weights(self.initial_weights)

                indices = np.random.choice(range(len(self.train_image)),
                                           data_volume,
                                           replace=False)
                x_train_sampled_image = self.train_image[indices]
                x_train_sampled_sequences = self.train_sequences[indices]
                y_train_sampled_label = self.train_label[indices]

                custom_callback = self.CustomCallback(epochs)

                history = self.model.fit(
                    [x_train_sampled_image, x_train_sampled_sequences],
                    y_train_sampled_label,
                    validation_data=([self.test_image, self.test_sequences], self.test_label),
                    epochs=epochs, batch_size=32, verbose=0, callbacks=[custom_callback]
                )
                history_list.append(history)
                predictions = self.model.predict([self.test_image, self.test_sequences])
                candidates = self.convert_predictions_to_text(predictions)
                references = self.convert_label_to_text(self.test_label)
                bleu_score = self.calculate_bleu(references, candidates)
                self.predictions_dict[(epochs, data_volume)] = predictions
                self.predictions_text_dict[(epochs, data_volume)] = candidates
                self.bleu_score_dict[(epochs, data_volume)] = bleu_score

            self.history_dict[f'history_epochs_{epochs}'] = history_list

        return (self.history_dict,
                self.predictions_dict,
                self.predictions_text_dict,
                self.test_label,
                self.bleu_score_dict)

def save_predictions_to_hdf5(predictions_dict, filename_h5):
    with h5py.File(filename_h5, 'w') as f:
        for key, predictions in predictions_dict.items():
            if not isinstance(predictions, np.ndaray):
                predictions = np.array(predictions)

            f.create_dataset(str(key), data=predictions)

def save_history_dict_to_hdf5(history_dict, filename_h5):
    with h5py.File(filename_h5, 'w') as f:
        for key, history_list in history_dict.items():
            for i, history in enumerate(history_list):
                for metric, values in history.history.items():
                    dataset_name = f'{key}/{i}/{metric}'
                    f.create_dataset(dataset_name, data=np.array(values))

def save_test_label_to_hdf5(test_label, filename_h5):
    with h5py.File(filename_h5, 'w') as f:
        if not isinstance(test_label, np.ndarray):
            test_label = np.array(test_label)
        f.create_dataset('test_label', data=test_label)

def save_bley_to_hdf5(bleu_score_dict, filename_h5):
    with h5py.File(filename_h5, 'w') as f:
        for key, bleu in bleu_score_dict.items():
            if not isinstance(bleu, np.ndarray):
                bleu = np.array(bleu)
            f.create_dataset(str(key), data=bleu)

def save_predictions_text_to_json(predictions_text_dict, filename_json):
    converted_dict = {f'{key[0]}_{key[1]}': value for key, value in predictions_text_dict.items()}
    with open(filename_json, 'w') as file:
        json.dump(converted_dict, file)

def save_tokenizer_to_json(tokenizer, filename_json):
    tokenizer_json = tokenizer.to_json()
    with open(filename_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def display_image_with_caption(test_image, predictions, actual_texts, index):
    image = test_image[index]
    predicted_caption = predictions[index]
    actual_caption = actual_texts[index]

    plt.imshow(image)
    plt.title(f"predicted: {predicted_caption}\n actual: {actual_caption}")
    plt.show()


def scores_to_df(history_dict, bleu_score_dict, train_data_volume):
    acc_scores, loss_scores, val_acc_scores, val_loss_scores = {}, {}, {}, {}
    score_data = []

    for epochs_key, histories in history_dict.items():
        epochs = int(epochs_key.split('_')[-1])
        for i, history in enumerate(histories):
            data_volume = train_data_volume[i]
            acc_scores[(epochs, data_volume)] = history.history['accuracy'][-1]
            loss_scores[(epochs, data_volume)] = history.history['loss'][-1]
            val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
            val_acc_scores[(epochs, data_volume)] = history.history[val_acc_key][-1]
            val_loss_scores[(epochs, data_volume)] = history.history['val_loss'][-1]

    for key in acc_scores.keys():
        epochs, data_volume = key
        acc = acc_scores[key]
        loss = loss_scores[key]
        val_acc = val_acc_scores[key]
        val_loss = val_loss_scores[key]
        score_data.append([epochs, data_volume, acc, loss, val_acc, val_loss])
    df_metrics_scores = pd.DataFrame(score_data,
                                     columns=['epochs', 'data_volume',
                                              'acc', 'loss',
                                              'val_acc', 'val_loss'])

    bleu_dict = {f'{key[0]}_{key[1]}': value for key, value in bleu_score_dict.items()}
    epoch_loop = []
    data_loop = []
    bleu = []

    for key, value in bleu_dict.items():
        params = key.split('_')
        epoch = params[0]
        data_volume = params[1]
        score = value

        epoch_loop.append(epoch)
        data_loop.append(data_volume)
        bleu.append(score)

    df_bleu_score = pd.DataFrame({'epochs':epoch_loop,
                                  'data_volume':data_loop,
                                  'bleu_score':bleu})

    df_metrics_scores['bleu_score'] = df_bleu_score['bleu_score']
    df_scores = df_metrics_scores
    return df_scores


def scores_to_plot(df_scores, image_name_jpg):
    df = df_scores
    epochs_list = df['epochs'].unique()
    data_volume_list = df['data_volume'].unique()

    accuracy = []
    loss =[]
    val_accuracy = []
    val_loss = []
    bleu_score = []

    for i in epochs_list:
        acc = df[df['accuracy'] == i]
        lss = df[df['loss'] == i]
        val_acc = df[df['val_acc'] == i]
        val_lss = df[df['val_loss'] == i]
        bleu = df[df['bleu_score'] == i]
        accuracy.append(acc)
        loss.append(lss)
        val_accuracy.append(val_acc)
        val_loss.append(val_lss)
        bleu_score.append(bleu)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    for epoch in range(len(epochs_list)):
        for acc in range(len(accuracy)):
            axes[0, 0].plot(data_volume_list, accuracy[acc], label=f"acc_{epochs_list[epoch]}", marker='o')

        for lss in range(len(loss)):
            axes[1, 0].plot(data_volume_list, loss[lss], label=f"loss_{epochs_list[epoch]}", marker='o')

        for val_acc in range(len(val_accuracy)):
            axes[0, 1].plot(data_volume_list, val_accuracy[val_acc], label=f"val_acc_{epochs_list[epoch]}", marker='o')

        for val_lss in range(len(val_loss)):
            axes[1, 1].plot(data_volume_list, val_loss[val_lss], label=f"val_loss_{epochs_list[epoch]}", marker='o')

        for bleu in range(len(bleu_score)):
            axes[0, 2].plot(data_volume_list, bleu_score[bleu], label=f"bleu_score_{epochs_list[epoch]}", marker='o')

    axes[0, 0].set_title('Training Accuracy')
    axes[0, 0].set_xlabel('Data volume')
    axes[0, 0].set_ylabel('Scores')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(loc='center right')

    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Data volume')
    axes[0, 1].set_ylabel('Scores')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(loc='center right')

    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Data volume')
    axes[1, 0].set_ylabel('Scores')
    axes[1, 0].set_ylim(0, 4)
    axes[1, 0].legend(loc='center right')

    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].set_xlabel('Data volume')
    axes[1, 1].set_ylabel('Scores')
    axes[1, 1].set_ylim(0, 6)
    axes[1, 1].legend(loc='center right')

    axes[0, 2].set_title('BLEU score')
    axes[0, 2].set_xlabel('Data volume')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend(loc='center right')

    plt.tight_layout()
    plt.savefig(image_name_jpg, format='jpg', dpi=300)
    plt.show()
