
import streamlit as st
import streamlit.components.v1 as components
import streamlit_image_select


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
    page_title="Catalog",
    layout="wide"
)

st.markdown(f'<p style="background-color:#efffff;color:#153944;font-size:50px;border-radius:2%;">Vogue Harmony</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:darkblue;font-size:16px;">Your Style, Perfected: Unleash the Power of Compatibility</p>', unsafe_allow_html=True)


"""
------------------------------------------------------------------------------------
"""





cwdir=os.getcwd()
_success_text_upload = "File was uploaded successfully!"



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




# Category list
image_pth_catalog = 'Catalog'
_filelocation = os.path.join(cwdir, image_pth_catalog)
_category_list = [ 'bag', 'shoes', 'top', 'outwear', 'pants', 'eyewear', 'earrings', 'watches', 'hats',
                   'rings',  'bracelet', 'necklace',  'dress',  'skirt']

# Category Folder
df_category_all = {}
i=0
for cat in _category_list:
    _filelocation = os.path.join(cwdir, image_pth_catalog, cat)
    df_category = pd.DataFrame(columns=['image', 'path'])

    for _folder in [_filelocation]:
        imglist_category = os.listdir(_folder)
        for _img in imglist_category:
            data = {'image': [_img], 'path': [_folder]}
            df_category = pd.concat([df_category, pd.DataFrame(data)])

    df_category_all[i] = df_category.head(24)
    i=i+1





# Catalog
@st.cache_data(experimental_allow_widgets=True)
def show_catalog(df_category_all, category_input,_label):

    df_temp_ = df_category_all[category_input].copy(deep=True)
    df_temp_['image_path'] = df_temp_['path'] + '/' + df_temp_['image']
    _img_file_ = list(df_temp_['image_path'].unique())
    img = streamlit_image_select.image_select(label=_label, images=_img_file_, use_container_width=False)
    # st.image(img)
    return img





col0, col1 = st.columns(2)

user_selected_image_list = ['']*8


# st.markdown(f'<br><br><br><br>', unsafe_allow_html=True)
browseText = '<p style="background-color:white;color:black; font-size: 20px;"><b>Browse Catalog</b></p>'
st.markdown(browseText, unsafe_allow_html=True)

st.markdown(
    f'<p style="background-color:orange;font-size:2px;"><b><a style= color: orange" href="/Catalog" target="_self" >-</a></b></p></center>',
    unsafe_allow_html=True)



tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(_category_list[:8])

st.session_state['category_list'] = _category_list[:8]

# with col0:


with tab1:
    # img1 = show_catalog(df_category_all, 0, "Choose a {} from existing catalog".format(_category_list[0]))
    label1 = "Choose a {} from existing catalog".format(_category_list[0])
    df_temp_ = df_category_all[0].copy(deep=True)
    df_temp_['image_path'] = df_temp_['path'] + '/' + df_temp_['image']
    _img_file_ = list(df_temp_['image_path'].unique())
    img1 = streamlit_image_select.image_select(label=label1, images=_img_file_, use_container_width=False)
    # user_selected_image_list[0] = img1

with tab2:
    # img2 = show_catalog(df_category_all, 1, "Choose a {} from existing catalog".format(_category_list[1]))
    label2 = "Choose a {} from existing catalog".format(_category_list[1])
    df_temp_ = df_category_all[1].copy(deep=True)
    df_temp_['image_path'] = df_temp_['path'] + '/' + df_temp_['image']
    _img_file_ = list(df_temp_['image_path'].unique())
    img2 = streamlit_image_select.image_select(label=label2, images=_img_file_, use_container_width=False)
    # user_selected_image_list[1] = img2

with tab3:
    img3 = show_catalog(df_category_all, 2, "Choose a {} from existing catalog".format(_category_list[2]))
    # user_selected_image_list[2] = img3

with tab4:
    img4 = show_catalog(df_category_all, 3, "Choose an {} wear from existing catalog".format(_category_list[3]))
    # user_selected_image_list[3] = img4

with tab5:
    img5 = show_catalog(df_category_all, 4, "Choose a {} from existing catalog".format(_category_list[4]))
    # user_selected_image_list[4] = img5

with tab6:
    img6 = show_catalog(df_category_all, 5, "Choose an {} wear from existing catalog".format(_category_list[5]))
    # user_selected_image_list[5] = img6

with tab7:
    img7 = show_catalog(df_category_all, 6, "Choose an {} ring from existing catalog".format(_category_list[6]))
    # user_selected_image_list[6] = img7

with tab8:
    img8 = show_catalog(df_category_all, 7, "Choose a {} from existing catalog".format(_category_list[7]))
    # user_selected_image_list[7] = img8



user_selected_image_list = [ img1, img2, img3, img4, img5, img6, img7, img8 ]
# st.session_state['selected_images'] = user_selected_image_list


# with col1:

col_, row_ = 2, 4
fig_1, axes_1 = plt.subplots(col_, row_, figsize=(4, 4))
plt.rcParams["figure.autolayout"] = True

_size=100

i = 0
for pos_col in range(col_):
    for pos_row in range(row_):
        axes_1[pos_col][pos_row].imshow(Image.open(user_selected_image_list[i]).resize((_size, _size)))
        axes_1[pos_col][pos_row].set_title(str(_category_list[i]))
        axes_1[pos_col][pos_row].axis('off')
        i = i+1


snapshot_text = '<center><p style=color:darkblue; font-size: 10px;">outfit selection snapshot</p></center>'
st.sidebar.markdown(snapshot_text, unsafe_allow_html=True)

st.sidebar.pyplot(fig_1)


# with col1:

# col_, row_ = 1, 8
# fig_1, axes_1 = plt.subplots(col_, row_, figsize=(20, 3))
# plt.rcParams["figure.autolayout"] = True
#
# _size=100
#
# i = 0
# for pos_col in range(row_):
#     axes_1[pos_col].imshow(mpimg.imread(user_selected_image_list[i]))
#     # axes_1[pos_col].set_title(str(_category_list[i]))
#     # axes_1[pos_col].axis('off')
#     axes_1[pos_col].set_xticklabels([])
#     axes_1[pos_col].set_yticklabels([])
#
#     axes_1[pos_col].set_xticks([])
#     axes_1[pos_col].set_yticks([])
#
#     axes_1[pos_col].spines['bottom'].set_color('grey')
#     axes_1[pos_col].spines['top'].set_color('grey')
#     axes_1[pos_col].spines['left'].set_color('grey')
#     axes_1[pos_col].spines['right'].set_color('grey')
#     i = i+1


# snapshot_text = '<center><p style=color:darkblue; font-size: 10px;">outfit selection snapshot</p></center>'
# snapshot_text = '<p style="background-color:#efffff;color:darkblue; font-size: 14px;">The selected outfit items which show up here </p>'
# st.sidebar.markdown(snapshot_text, unsafe_allow_html=True)
#
# st.sidebar.pyplot(fig_1)







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

    blank_image_array = np.zeros((250, 250, 3), dtype=np.uint8)
    blank_image = Image.fromarray(blank_image_array)
    blank_image = blank_image.resize((250, 250), Image.LANCZOS)

    for index, row in dataset.iterrows():
        outfit_data = []

        for item_count in range(1, 9):
            try:
                image_path = dataset['item_image_' + str(item_count)].unique()[0]
                image = Image.open(image_path).convert(rpg_slash_const)
                image = image.resize((250, 250), Image.LANCZOS)
                valid_image_count = valid_image_count + 1
            except:
                image = blank_image
                dummy_image_count = dummy_image_count + 1
            datu = np.asarray(image)
            normu_dat = datu / 255
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
    filepath1 = os.path.join(cwdir, 'model/model_compatibility_score')

    modelFileList = []
    for modelFile in os.listdir(filepath1):
        if modelFile.find('h5') > -1:
            modelFileList.append(modelFile)

    modelFileList = sorted(modelFileList)
    modelFile = modelFileList[-1]

    model_CNN_LSTM = load_model(os.path.join(filepath1, modelFile))

    # Predict Compatibility
    predicted_probability = model_CNN_LSTM.predict(df_unseen_data, verbose=0)

    predicted_score = np.argmax(predicted_probability)

    # st.write("Probability : ", predicted_probability)
    # st.write("Predicted Score : ", predicted_score)

    return predicted_score




# score = CNN_LSTM_score_prediction(user_selected_image_list)

uploaded_cnt = 0

if uploaded_cnt == 0:

    rating = random.randint(1, 4)
    # rating = score
    compatibility = 'NA'

    rating = 1

    if rating == 0:
        compatibility = 'Low compatibility'
        _status_color = 'red'
        _nextstep = 'You may consider modifying the outfit'

    elif rating <= 1:
        compatibility = 'Medium compatibility'
        _status_color = 'darkblue'
        _nextstep = 'You are okay to proceed'

    else:
        compatibility = 'High compatibility'
        _status_color = 'green'
        _nextstep = 'Please Proceed to Buy'


    # st.write("Status: ", compatibility)

    # col01, col02 = st.columns(2)
    #
    # with col01:
    # compatibility_text = '<p style="color:DarkSlateBlue; font-size: 12px;"></p>'
    # st.markdown(compatibility_text, unsafe_allow_html=True)

    st.sidebar.markdown(
        f'<center><p style="background-color:{_status_color};color:white;font-size:12px;"> {compatibility}. {_nextstep} </p></center>'.format(
            _status_color, compatibility, _nextstep), unsafe_allow_html=True)

    # with col02:
    #     st.markdown(
    #         f'<center><p style="background-color:orange;color:white;font-size:14px;"><b><a style= "text-decoration: none;color: white" href="http://localhost:8502/Best_Match" target="_self" >Match an item with current outfit</a></b></p></center>',
    #         unsafe_allow_html=True)
    #


else:
    st.write('Please Upload all 5 images!')





# Below is the New Item Suggestion section






# 9th Image
df_temp = df_predict.copy(deep=True)
df_temp['image_path'] = df_temp['path'] + '/' + df_temp['image']
_img_file = list(df_temp['image_path'].unique())
_img_file = random.sample(_img_file, len(_img_file))
_img_file = _img_file[0]


"""
------------------------------------------------------------------------------------
"""

# st.markdown(f'<br><br><br><br>', unsafe_allow_html=True)
# browseText = '<p style="background-color:white;color:black; font-size: 20px;"><b>New Item Recommendation</b></p>'
# st.markdown(browseText, unsafe_allow_html=True)
#
# st.markdown(
#     f'<p style="background-color:orange;font-size:2px;"><b><a style= color: orange" href="" target="_self" >-</a></b></p></center>',
#     unsafe_allow_html=True)
#
# text_upload = '<p style="background-color:#ffffef;color:darkblue; font-size: 18px;">We have Completed your outfit</p>'
# st.markdown(text_upload, unsafe_allow_html=True)



col_, row_ = 5, 2
fig_reco, axes_ = plt.subplots(row_, col_, figsize = (16, 5))
# plt.rcParams["figure.autolayout"] = True

placeholder_image = os.path.join(df_placeholder.path.unique()[0], df_placeholder.image.unique()[0])


def highlight_predicted_image(val_i, pos_row, pos_col):

    global axes_
    global _img_file

    axes_[pos_row][pos_col].imshow(mpimg.imread(_img_file))
    axes_[pos_row][pos_col].set_title('outfit ' + str(val_i+1))

    axes_[pos_row][pos_col].set_xticklabels([])
    axes_[pos_row][pos_col].set_yticklabels([])

    axes_[pos_row][pos_col].set_xticks([])
    axes_[pos_row][pos_col].set_yticks([])

    axes_[pos_row][pos_col].spines['bottom'].set_color('red')
    axes_[pos_row][pos_col].spines['top'].set_color('red')
    axes_[pos_row][pos_col].spines['left'].set_color('red')
    axes_[pos_row][pos_col].spines['right'].set_color('red')




i = 0
for pos_row in range(row_):
    for pos_col in range(col_):
        try:
            axes_[pos_row][pos_col].imshow(Image.open(user_selected_image_list[i]))
            axes_[pos_row][pos_col].set_title(str(_category_list[i]))
            axes_[pos_row][pos_col].axis('off')

        except:

            if i == 8:
                highlight_predicted_image(i, pos_row, pos_col)
            else:
                axes_[pos_row][pos_col].imshow(Image.open(placeholder_image))
                axes_[pos_row][pos_col].set_title('')
                axes_[pos_row][pos_col].axis('off')

        i = i + 1


col000, col001 = st.columns(2)

with col000:
    st.pyplot(fig_reco)



"""
------------------------------------------------------------------------------------
"""

