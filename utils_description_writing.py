from raw_to_csv import convert_json_to_csv
import pandas as pd
from IPython.display import Image, display
from ipywidgets import widgets, Layout, interactive
from IPython.display import clear_output
import numpy as np
import re

"""
utils for writing description
final output: label data for train(with description sentences)
this module will not be used in main.py
    label for model train csv will be directly loaded as "train_sample_2000.csv"
"""

# prepare label data(without description)
folder_path = './label_raw'
df = convert_json_to_csv(folder_path)

# define writing works class
########################################################################
class description_write:
    def __init__(self, dataframe, data_name, img_dir, start_index):
        self.data = dataframe
        self.data_name = data_name
        self.img_dir = img_dir
        self.current_instance_idx = start_index
        self.text_input = widgets.Text(
            placeholder="write the description",
            layout=Layout(width="50%")
            )
        self.output = widgets.Output()
        self.text_input.on_submit(self.handle_enter_press)
        self.next_button = widgets.Button(description="Next Instance")

        self.image_mapping = {}
        for image_id in self.data['file_name']:
            self.image_mapping[image_id] = image_id

        self.text_input.observe(self.on_text_input_change, names='value')
        self.next_button.on_click(self.next_instance)

        self.interactive_widget = interactive(self.display_current_instance,
                                              instance_idx=widgets.IntSlider(min=0,
                                                                             max=len(self.data) - 1))
        display(self.interactive_widget)

    def show_image(self, image_id):
        image_filename = f"{self.img_dir}/{image_id}"
        display(Image(filename=image_filename, width=400, height=400))

    def display_current_instance(self, instance_idx):

        image_id = self.data.iloc[instance_idx]['file_name']
        id = self.data.iloc[instance_idx]['id']

        self.show_image(image_id)
        print('id: ', id)
        print('file_name: ', image_id)

        with self.output:
            clear_output()

        display(self.text_input)
        display(self.next_button)

        self.current_instance_idx = instance_idx

    def on_text_input_change(self, change):
        if change['new']:
            input_text = change['new']
            self.data.at[self.current_instance_idx, 'sentence_en'] = input_text

    def handle_enter_press(self, text_input):
        if self.text_input.value == '!save' or self.current_instance_idx == len(self.data) - 1:
            self.save_and_close()
        else:
            self.next_instance(None)

    def save_and_close(self):
        self.text_input.close()
        self.output.close()
        self.next_button.close()
        self.data.to_csv(f"{self.data_name}{self.current_instance_idx}.csv", encoding='cp949', index=False)
        print(f"{self.data_name}{self.current_instance_idx}.csv saved")

    def next_instance(self, b):
        if self.current_instance_idx < len(self.data) - 1:
            self.current_instance_idx += 1
            self.interactive_widget.children[0].value = self.current_instance_idx
            self.text_input.value = ""
        else:
            self.save_and_close()

#######################################################################################
# activate class and doing description writing works

img_dir = '/train_image'
start_index = 0
description_writer = description_write(df, 'description', img_dir, start_index)