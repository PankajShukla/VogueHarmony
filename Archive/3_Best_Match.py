
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



st.set_page_config(
    page_title="BestMatch",
    layout="wide"
)

st.markdown(f'<p style="background-color:#efffff;color:#153944;font-size:50px;border-radius:2%;">Vogue Harmony</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:darkblue;font-size:16px;">Your Style, Perfected: Unleash the Power of Compatibility</p>', unsafe_allow_html=True)




cwdir=os.getcwd()


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


# 9th Image
df_temp = df_predict.copy(deep=True)
df_temp['image_path'] = df_temp['path'] + '/' + df_temp['image']
_img_file = list(df_temp['image_path'].unique())
_img_file = random.sample(_img_file, len(_img_file))
_img_file = _img_file[0]



user_selected_image_list = st.session_state['selected_images']
_category_list = st.session_state['category_list']


"""
------------------------------------------------------------------------------------
"""

text_upload = '<center><p style="background-color:#ffffef;color:darkblue; font-size: 18px;">We have Completed your outfit</p></center>'
st.markdown(text_upload, unsafe_allow_html=True)



col_, row_ = 5, 2
fig_, axes_ = plt.subplots(row_, col_, figsize = (16, 5))
plt.rcParams["figure.autolayout"] = True

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
            axes_[pos_row][pos_col].imshow(mpimg.imread(user_selected_image_list[i]))
            axes_[pos_row][pos_col].set_title(str(_category_list[i]))
            axes_[pos_row][pos_col].axis('off')

        except:

            if i == 8:
                highlight_predicted_image(i, pos_row, pos_col)
            else:
                axes_[pos_row][pos_col].imshow(mpimg.imread(placeholder_image))
                axes_[pos_row][pos_col].set_title('')
                axes_[pos_row][pos_col].axis('off')

        i = i + 1



st.pyplot(fig_)



"""
------------------------------------------------------------------------------------
"""

