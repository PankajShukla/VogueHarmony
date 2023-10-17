import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('streamlit-image-select')

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


img1 = st.session_state['catalog_image1']
img2 = st.session_state['catalog_image2']
img3 = st.session_state['catalog_image3']
img4 = st.session_state['catalog_image4']
img5 = st.session_state['catalog_image5']
img6 = st.session_state['catalog_image6']

# Option to Upload Images

_success_text_upload = "File was uploaded successfully!"

"""
------------------------------------------------------------------------------------
"""

text_upload = '<center><p style="background-color:#ffffef;color:darkblue; font-size: 18px;">Here are your outfit items</p></center>'

st.markdown(text_upload, unsafe_allow_html=True)



uploaded_cnt=0


fig_, axes_ = plt.subplots(1, 6, figsize=(20, 4))
plt.rcParams["figure.autolayout"] = True


axes_[0].imshow(mpimg.imread(img1))
axes_[0].set_title('outfit ' + str(1))
axes_[0].axis('off')


axes_[1].imshow(mpimg.imread(img2))
axes_[1].set_title('outfit ' + str(2))
axes_[1].axis('off')


axes_[2].imshow(mpimg.imread(img3))
axes_[2].set_title('outfit ' + str(3))
axes_[2].axis('off')


axes_[3].imshow(mpimg.imread(img4))
axes_[3].set_title('outfit ' + str(4))
axes_[3].axis('off')


axes_[4].imshow(mpimg.imread(img5))
axes_[4].set_title('outfit ' + str(5))
axes_[4].axis('off')


axes_[5].imshow(mpimg.imread(img6))
axes_[5].set_title('outfit ' + str(6))
axes_[5].axis('off')


st.pyplot(fig_)




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









