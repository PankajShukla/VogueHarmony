
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

import h5py

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.layers import TimeDistributed
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPool2D
from keras.layers import GRU, Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.models import load_model


st.set_page_config(
    page_title="Home",
    layout="wide"
)

st.markdown(f'<p style="background-color:#efffff;color:#153944;font-size:50px;border-radius:2%;">Vogue Harmony</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:darkblue;font-size:16px;">Your Style, Perfected: Unleash the Power of Compatibility</p>', unsafe_allow_html=True)




m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
</style>""", unsafe_allow_html=True)



"""
------------------------------------------------------------------------------------
"""



# Upload Folder
cwdir=os.getcwd()
image_pth = 'icon'
_filelocation = os.path.join(cwdir, image_pth)


gif_list = ["https://giphy.com/embed/hHPfqPaflVT41Nr5qJ/video",
            "https://giphy.com/embed/lW8XxDIKEIxb4gKbbq/video",
            "https://giphy.com/embed/9Br9RJd4eugYQoIdob/video",
            "https://giphy.com/embed/t4yJm5CrafgyQfZ04T/video",
           ]


gif_list = random.sample(gif_list, len(gif_list))
gif_list = gif_list[:3]

col1, col2, col3 = st.columns(3)

col1.markdown("""<html><body><iframe src={gif} width="500" height="300" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></body></html>""".format(gif=gif_list[0]), unsafe_allow_html=True)
col2.markdown("""<html><body><iframe src={gif} width="500" height="300" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></body></html>""".format(gif=gif_list[1]), unsafe_allow_html=True)
col3.markdown("""<html><body><iframe src={gif} width="500" height="300" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></body></html>""".format(gif=gif_list[2]), unsafe_allow_html=True)




"""

------------------------------------------------------------------------------------
"""


st.markdown(f'<center><p style="background-color:#ec4c8c;color:white;font-size:20px;"><b> 3 Simple steps to get the best outfit for yourself </b>  </p></center>',
            unsafe_allow_html=True)



icon1 = os.path.join(_filelocation, 'catalog.png')
icon2 = os.path.join(_filelocation, 'compatibility.png')
icon3 = os.path.join(_filelocation, 'recommendation.png')
icon4 = os.path.join(_filelocation, 'blank.png')


fig_icon, axes_icon = plt.subplots(1, 5, figsize=(2.5, 10))
plt.rcParams["figure.autolayout"] = True


col21, col22, col23, col24, col25 =  st.columns(5)

_icon_size = 250

with col21:
    axes_icon[0].imshow(Image.open(icon4))
    axes_icon[0].axis('off')

with col22:
    axes_icon[1].imshow(Image.open(icon1).resize((_icon_size, _icon_size)))
    axes_icon[1].axis('off')

with col23:
    axes_icon[2].imshow(Image.open(icon2).resize((_icon_size, _icon_size)))
    axes_icon[2].axis('off')

with col24:
    axes_icon[3].imshow(Image.open(icon3).resize((_icon_size, _icon_size)))
    axes_icon[3].axis('off')

with col25:
    axes_icon[4].imshow(Image.open(icon4))
    axes_icon[4].axis('off')





st.pyplot(fig_icon)


col11, col12, col13, col14, col15 =  st.columns(5)

with col12:
    st.markdown( f'<center><p style="color:darkblue;font-size:16px;"><b>Browse our Catalog</b></p></center>',unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:blue;font-size:16px;"> <center>Choose products from our catalog across different '
        f''
        f'categories to create an outfit</center></p>',
        unsafe_allow_html=True)

with col13:
    st.markdown( f'<center><p style="color:darkblue;font-size:16px;"><b>Get Compatability Score</b></p></center>',unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:blue;font-size:16px;"><center>Check Compatability of the outfit items within the outfit '
        f''
        f'using our recommendation system</center></p>',
        unsafe_allow_html=True)


with col14:
    st.markdown( f'<center><p style="color:darkblue;font-size:16px;"><b>Pick the perfect match</b></p></center>',unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:blue;font-size:16px;"><center>Purchase the best outfit item '
        f''
        f'from the recommended list</center></p>',
        unsafe_allow_html=True)


st.markdown(f'<br><br>', unsafe_allow_html=True)
st.markdown(f'<center><p style="background-color:#ec4c8c;color:white;font-size:20px;"><b> Catalog and Compatibility </b>  </p></center>',
            unsafe_allow_html=True)
st.markdown(f'<br>', unsafe_allow_html=True)


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

placeholder_image = os.path.join(df_placeholder.path.unique()[0], df_placeholder.image.unique()[0])


# Default Catalog Folder

image_pth = 'DefaultCatalog'
_filelocation = os.path.join(cwdir, image_pth)

df_default = pd.DataFrame(columns=['image', 'path'])

for _folder in [_filelocation]:
    imglist_default = os.listdir(_folder)
    print(_folder, ": ", len(imglist_default))
    for _img in imglist_default:
        data = {'image': [_img], 'path': [_folder]}
        df_default = pd.concat([df_default, pd.DataFrame(data)])


default_catalog_image = os.path.join(df_default.path.unique()[0], df_default.image.unique()[0])




# Black Image Folder for Model Input

image_pth = 'BlackImage'
_filelocation = os.path.join(cwdir, image_pth)

df_black = pd.DataFrame(columns=['image', 'path'])

for _folder in [_filelocation]:
    imglist_black = os.listdir(_folder)
    print(_folder, ": ", len(imglist_black))
    for _img in imglist_black:
        data = {'image': [_img], 'path': [_folder]}
        df_black = pd.concat([df_black, pd.DataFrame(data)])


black_image = os.path.join(df_black.path.unique()[0], df_black.image.unique()[0])



# Category list
image_pth_catalog = 'Catalog'
_filelocation = os.path.join(cwdir, image_pth_catalog)
_category_list = [ 'Bag', 'Shoe', 'Top', 'Outwear', 'Pant', 'Eyewear', 'Earring', 'Watch', 'Hat',
                   'Ring',  'Bracelet', 'Necklace',  'Dress',  'Skirt']




# Category Folder
df_category_all = {}
num_of_images_in_each_catalog=28
i=0
for cat in _category_list:
    _filelocation = os.path.join(cwdir, image_pth_catalog, cat)
    df_category = pd.DataFrame(columns=['image', 'path'])
    df_category = pd.concat([df_category, df_default])
    for _folder in [_filelocation]:
        imglist_category = os.listdir(_folder)
        for _img in imglist_category:
            data = {'image': [_img], 'path': [_folder]}
            df_category = pd.concat([df_category, pd.DataFrame(data)])

    df_category_all[i] = df_category.head(num_of_images_in_each_catalog)
    i=i+1




# Predicted Category Folder
df_category_all_predicted = {}
num_of_images_in_each_catalog_predicted = 4
i=0
for cat in _category_list:
    _filelocation = os.path.join(cwdir, image_pth_catalog, cat)
    df_category_predicted = pd.DataFrame(columns=['image', 'path'])
    for _folder in [_filelocation]:
        imglist_category = os.listdir(_folder)
        for _img in imglist_category:
            data = {'image': [_img], 'path': [_folder]}
            df_category_predicted = pd.concat([df_category_predicted, pd.DataFrame(data)])

    df_category_all_predicted[i] = df_category_predicted.tail(num_of_images_in_each_catalog_predicted)
    i=i+1



# Catalog

def show_catalog(category_input,_label):

    df_temp_ = df_category_all[category_input].copy(deep=True)
    df_temp_['image_path'] = df_temp_['path'] + '/' + df_temp_['image']
    _img_file_catalog = list(df_temp_['image_path'].unique())

    img = image_select(
        label=_label,
        images=_img_file_catalog,
        use_container_width=False
    )
    return img


def show_predicted(category_input, _label):

    df_temp_ = df_category_all_predicted[category_input].copy(deep=True)
    df_temp_['image_path'] = df_temp_['path'] + '/' + df_temp_['image']
    _img_file_predicted = list(df_temp_['image_path'].unique())[:4]

    img = image_select(
        label=_label,
        images=_img_file_predicted,
        use_container_width=False
    )
    return img






user_selected_image_list = ['']*8

col0, col1 = st.columns(2)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(_category_list[:8])

with col0:

    st.markdown('<br><br><br><br><br>', unsafe_allow_html=True)
    browseText = '<p style="background-color:white;color:black; font-size: 20px;"><b>Browse Catalog</b></p>'
    st.markdown(browseText, unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#ec4c8c;font-size:2px;"><b><a style= color: #daa520" href="/Catalog" target="_self" >-</a></b></p></center>', unsafe_allow_html=True)

    st.session_state['category_list'] = _category_list[:8]

    with tab1:
        img1 = show_catalog(0, "Choose a bag from existing catalog")
        if img1 == default_catalog_image:
            img1 = placeholder_image
        user_selected_image_list[0] = img1

    with tab2:
        img2 = show_catalog(1, "Choose a shoe from existing catalog")
        if img2 == default_catalog_image:
            img2 = placeholder_image
        user_selected_image_list[1] = img2

    with tab3:
        img3 = show_catalog(2, "Choose a top from existing catalog")
        if img3 == default_catalog_image:
            img3 = placeholder_image
        user_selected_image_list[2] = img3

    with tab4:
        img4 = show_catalog(3, "Choose an outer wear from existing catalog")
        if img4 == default_catalog_image:
            img4 = placeholder_image
        user_selected_image_list[3] = img4

    with tab5:
        img5 = show_catalog(4, "Choose a pants from existing catalog")
        if img5 == default_catalog_image:
            img5 = placeholder_image
        user_selected_image_list[4] = img5

    with tab6:
        img6 = show_catalog(5, "Choose an eye wear from existing catalog")
        if img6 == default_catalog_image:
            img6 = placeholder_image
        user_selected_image_list[5] = img6

    with tab7:
        img7 = show_catalog(6, "Choose an ear ring from existing catalog")
        if img7 == default_catalog_image:
            img7 = placeholder_image
        user_selected_image_list[6] = img7

    with tab8:
        img8 = show_catalog(7, "Choose a watch from existing catalog")
        if img8 == default_catalog_image:
            img8 = placeholder_image
        user_selected_image_list[7] = img8



    st.session_state['catalog_images_list'] = user_selected_image_list

    model_input_selected_image_list = user_selected_image_list[:8]

    for i in range(len(model_input_selected_image_list)):
        if model_input_selected_image_list[i] == placeholder_image:
            model_input_selected_image_list[i] = black_image




with col1:

    col_, row_ = 1, 8
    fig_1, axes_1 = plt.subplots(col_, row_, figsize=(20, 3))
    plt.rcParams["figure.autolayout"] = True

    _size=100

    i = 0
    for pos_col in range(row_):
        axes_1[pos_col].imshow(mpimg.imread(user_selected_image_list[i]))
        axes_1[pos_col].set_xticklabels([])
        axes_1[pos_col].set_yticklabels([])

        axes_1[pos_col].set_xticks([])
        axes_1[pos_col].set_yticks([])

        axes_1[pos_col].spines['bottom'].set_color('grey')
        axes_1[pos_col].spines['top'].set_color('grey')
        axes_1[pos_col].spines['left'].set_color('grey')
        axes_1[pos_col].spines['right'].set_color('grey')
        i = i+1


    snapshot_text = '<p style="background-color:#efffff;color:darkblue; font-size: 14px;">The selected outfit items which show up here </p>'
    st.markdown(snapshot_text, unsafe_allow_html=True)
    st.pyplot(fig_1)




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



    def CNN_LSTM_score_prediction(model_input_selected_image_list):

        # Create Unseen data in input format
        df_unseen_data = convert_image_to_array(model_input_selected_image_list)

        # Load Model
        filepath1 = os.path.join(cwdir, 'model/model_compatibility_score')
        # st.write(filepath1)
        modelFileList = []
        for modelFile in os.listdir(filepath1):
            if modelFile.find('h5') > -1:
                modelFileList.append(modelFile)

        modelFileList = sorted(modelFileList)
        modelFile = modelFileList[-1]
        # st.write(os.path.join(filepath1, modelFile))

        model_CNN_LSTM = load_model(os.path.join(filepath1, modelFile))

        # Predict Compatibility
        predicted_probability = model_CNN_LSTM.predict(df_unseen_data, verbose=0)

        predicted_score = np.argmax(predicted_probability)

        df_probability = pd.DataFrame(predicted_probability)
        # st.write(df_probability)

        label0 = df_probability.iloc[0, 0]
        label1 = df_probability.iloc[0, 1]
        label2 = df_probability.iloc[0, 2]
        label3 = df_probability.iloc[0, 3]
        cumscore123 = label1 + label2 + label3
        cumscore12  = label1 + label2

        # st.write(label0, label1, label2, label2)

        if cumscore12 >= 0.3:
            predicted_score = 1
        elif cumscore12 >= 0.4:
            predicted_score = 2
        elif cumscore12 >= 0.4 and cumscore123 >= 0.5:
            predicted_score = 3
        else:
            predicted_score = 0

        # st.write("Probability : ", predicted_probability)
        # st.write("Predicted Score : ", predicted_score)

        return predicted_score




    score = CNN_LSTM_score_prediction(user_selected_image_list)
    rating = random.randint(1, 4)
    rating = score
    cat_num = (rating+1)*5

    uploaded_cnt = len(list(set(model_input_selected_image_list)))-1

    if uploaded_cnt >= 5 :

        compatibility = 'NA'

        if rating == 0:
            compatibility = 'Low compatibility'
            _status_color = 'red'
            _nextstep = 'Select another item.'

        elif rating <= 2:
            compatibility = 'Medium compatibility'
            _status_color = '#014f86'
            _nextstep = 'You are okay to proceed.'

        else:
            compatibility = 'High compatibility'
            _status_color = 'green'
            _nextstep = 'Please Proceed to Buy.'


        # st.write("Status: ", compatibility)


        col01, col02 = st.columns(2)

        with col01:

            compatibility_text = '<p style="color:DarkSlateBlue; font-size: 12px;"></p>'
            st.markdown(compatibility_text, unsafe_allow_html=True)

            st.markdown(
                f'<center><p style="background-color:{_status_color};color:white;font-size:12px;"> {compatibility}. {_nextstep} </p></center>'.format(
                    _status_color, compatibility, _nextstep), unsafe_allow_html=True)


    else:
        st.write('Please select atleast 8 outfit items to get compatibility score!')




def highlight_predicted_image(val_i, pos_col, axes_, _img_file):

    axes_[pos_col].imshow(mpimg.imread(_img_file))
    axes_[pos_col].set_title('recommended item')

    axes_[pos_col].set_xticklabels([])
    axes_[pos_col].set_yticklabels([])

    axes_[pos_col].set_xticks([])
    axes_[pos_col].set_yticks([])

    axes_[pos_col].spines['bottom'].set_color('red')
    axes_[pos_col].spines['top'].set_color('red')
    axes_[pos_col].spines['left'].set_color('red')
    axes_[pos_col].spines['right'].set_color('red')





def outfit_summary_and_purchase(_predicted_image):

    st.markdown(f'<br>', unsafe_allow_html=True)
    browseText = '<p style="background-color:white;color:black; font-size: 20px;"><b>Outfit Summary</b></p>'
    st.markdown(browseText, unsafe_allow_html=True)
    st.markdown(
        f'<p style="background-color:#ec4c8c;font-size:2px;"><b><a style= color: #daa520" href="" target="_self" >-</a></b></p></center>',
        unsafe_allow_html=True)

    col_, row_ = 9, 1
    fig_, axes_ = plt.subplots(row_, col_, figsize=(16, 4))
    plt.rcParams["figure.autolayout"] = True

    i = 0

    for pos_col in range(col_):
        try:
            axes_[pos_col].imshow(mpimg.imread(user_selected_image_list[i]))
            axes_[pos_col].set_title(str(_category_list[i]))
            # axes_[pos_col].axis('off')
            axes_[pos_col].set_xticklabels([])
            axes_[pos_col].set_yticklabels([])

            axes_[pos_col].set_xticks([])
            axes_[pos_col].set_yticks([])

            axes_[pos_col].spines['bottom'].set_color('lightgrey')
            axes_[pos_col].spines['top'].set_color('lightgrey')
            axes_[pos_col].spines['left'].set_color('lightgrey')
            axes_[pos_col].spines['right'].set_color('lightgrey')

        except:

            if i == 8:
                highlight_predicted_image(i, pos_col, axes_, _predicted_image)
            else:
                axes_[pos_col].imshow(mpimg.imread(placeholder_image))
                axes_[pos_col].set_title('')
                axes_[pos_col].axis('off')

        i = i + 1

    st.pyplot(fig_)

    st.markdown(f'<br>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="background-color:#ec4c8c;font-size:2px;"><b><a style= color: #daa520" href="" target="_self" >-</a></b></p></center>',
        unsafe_allow_html=True)

    col11, col12, col13, col14, col15, col16 = st.columns(6)
    with col11:
        st.markdown(
            f'<center><p style="background-color:#ec4c8c;color:white;font-size:18px;"> Proceed to Check Out </p></center>',
            unsafe_allow_html=True)




if uploaded_cnt >= 5:

    # Below is the New Item Suggestion section

    st.markdown(f'<br><br>', unsafe_allow_html=True)
    st.markdown(f'<center><p style="background-color:#ec4c8c;color:white;font-size:20px;"><b> Recommending a Best matching outfit item </b>  </p></center>', unsafe_allow_html=True)

    colpred1, colpred2 = st.columns(2)
    # with colpred1:
        # st.pyplot(fig_pred)
    new_item_image = show_predicted(cat_num, "We have recommendeded Best four items for you which may go well with the chosen outfit. Please Choose one. ".format(_category_list[cat_num]))

    predicted_image = new_item_image
    outfit_summary_and_purchase(predicted_image)





"""
------------------------------------------------------------------------------------
"""


