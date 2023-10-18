
import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_select import image_select

from PIL import Image
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import glob
import shutil



import numpy as np
import json
import math

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.layers import TimeDistributed
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPool2D
from keras.layers import GRU, Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.models import load_model


st.set_page_config(
    page_title="AI",
    layout="wide"
)

st.markdown(f'<p style="background-color:#efffff;color:#153944;font-size:50px;border-radius:2%;">Vogue Harmony</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:darkblue;font-size:16px;">Your Style, Perfected: Unleash the Power of Compatibility</p>', unsafe_allow_html=True)




cwdir=os.getcwd()


# Upload Folder
image_pth = 'Upload'
_filelocation = os.path.join(cwdir, image_pth)

df_upload = pd.DataFrame(columns=['image', 'path'])

for _folder in [_filelocation]:
    imglist_upload = os.listdir(_folder)
    for _img in imglist_upload:
        data = {'image': [_img], 'path': [_folder]}
        df_upload = pd.concat([df_upload, pd.DataFrame(data)])



# Predict Folder
image_pth = 'Predict'
_filelocation = os.path.join(cwdir, image_pth)

df_predict = pd.DataFrame(columns=['image', 'path'])

for _folder in [_filelocation]:
    imglist_predict = os.listdir(_folder)
    print(_folder, ": ", len(imglist_predict))
    for _img in imglist_predict:
        data = {'image': [_img], 'path': [_folder]}
        df_predict = pd.concat([df_predict, pd.DataFrame(data)])


# Placeholder Folder

image_pth = 'Placeholder'
_filelocation = os.path.join(cwdir, image_pth)

df_placeholder = pd.DataFrame(columns=['image', 'path'])

for _folder in [_filelocation]:
    imglist_placeholder = os.listdir(_folder)
    print(_folder, ": ", len(imglist_placeholder))
    for _img in imglist_placeholder:
        data = {'image': [_img], 'path': [_folder]}
        df_placeholder = pd.concat([df_placeholder, pd.DataFrame(data)])


user_selected_image_list = [st.session_state['catalog_image1'],
                            st.session_state['catalog_image2'],
                            st.session_state['catalog_image3'],
                            st.session_state['catalog_image4'],
                            st.session_state['catalog_image5'],
                            st.session_state['catalog_image6'],
                            st.session_state['catalog_image7'],
                            st.session_state['catalog_image8']
                            ]

# Option to Upload Images

_success_text_upload = "File was uploaded successfully!"

"""
------------------------------------------------------------------------------------
"""

text_upload = '<center><p style="background-color:#ffffef;color:darkblue; font-size: 18px;">Here are your outfit items</p></center>'

st.markdown(text_upload, unsafe_allow_html=True)



uploaded_cnt=0


fig_, axes_ = plt.subplots(1, 8, figsize=(20, 4))
plt.rcParams["figure.autolayout"] = True


for ax_i in range(8):
    axes_[ax_i].imshow(mpimg.imread(user_selected_image_list[ax_i]))
    axes_[ax_i].set_title('outfit ' + str(ax_i+1))
    axes_[ax_i].axis('off')

st.pyplot(fig_)




def convert_image_to_array(input_img_list):

    df_temp = {'item_image_1': [input_img_list[0]],
               'item_image_2': [input_img_list[1]],
               'item_image_3': [input_img_list[2]],
               'item_image_4': [input_img_list[3]],
               'item_image_5': [input_img_list[4]],
               'item_image_6': [input_img_list[5]],
               'item_image_7': [input_img_list[6]],
               'item_image_8': [input_img_list[7]]
               }

    dataset = pd.DataFrame(df_temp)

    outfit_list = []

    column_index = len(dataset.columns) - 1
    valid_image_count = 0
    dummy_image_count = 0
    rpg_slash_const = "RGB"

    blank_image_array = np.zeros((250, 250, 3), dtype = np.uint8)
    blank_image = Image.fromarray(blank_image_array)
    blank_image = blank_image.resize((250, 250), Image.LANCZOS)

    for index, row in dataset.iterrows():
        outfit_data = []

        for item_count in range(1, 9):
          try:
            image_path = dataset['item_image_'+str(item_count)].unique()[0]
            image = Image.open(image_path).convert(rpg_slash_const)
            image = image.resize((250, 250), Image.LANCZOS)
            valid_image_count = valid_image_count + 1
          except:
            image = blank_image
            dummy_image_count = dummy_image_count + 1
          datu = np.asarray(image)
          normu_dat = datu/255
          outfit_data.append(normu_dat)

        outfit_data = np.array(outfit_data)
        outfit_list.append(outfit_data)

    print("Num of real images = ", valid_image_count)
    print("Num of dummy images = ", dummy_image_count)

    return np.array(outfit_list)


def CNN_LSTM_score_prediction(user_selected_image_list):

    # Create Unseen data in input format
    df_unseen_data = convert_image_to_array(user_selected_image_list)

    # Load Model
    modelpath = 'model_compatibility_score/cnn_lstm_model_new3.h5'
    filepath1 = os.path.join(cwdir, 'model', modelpath)
    model_CNN_LSTM = load_model(filepath1)

    # Predict Compatibility
    predicted_probability = model_CNN_LSTM.predict(df_unseen_data, verbose=0)

    predicted_score = np.argmax(predicted_probability)

    print(predicted_probability)
    print(predicted_score)


    return predicted_score



# score = CNN_LSTM_score_prediction(user_selected_image_list)



if uploaded_cnt==0:

    rating = random.randint(1,10)
    compatibility = 'NA'

    if rating <= 4:
        compatibility ='Low compatibility'
        _status_color = 'red'
        _nextstep = 'You may consider modifying the outfit.'

    elif rating <= 7:
        compatibility ='Medium compatibility'
        _status_color = 'blue'
        _nextstep = 'You are okay to proceed.'

    else:
        compatibility ='High compatibility'
        _status_color = 'green'
        _nextstep = 'Please Proceed.'


    compatibility_text = '<p style="color:DarkSlateBlue; font-size: 18px;"></p>'
    st.markdown(compatibility_text, unsafe_allow_html=True)

    st.markdown(
        f'<center><p style="background-color:#ffffef;color:{_status_color};font-size:24px;"><b> The items in the outfit have {compatibility}. {_nextstep} </b>  </p></center>'.format(
            _status_color,compatibility,_nextstep),unsafe_allow_html=True)

else:
    st.write('Please Upload all 5 images!')

