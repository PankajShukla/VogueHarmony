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
    page_title="Catalog",
    layout="wide"
)

st.markdown(f'<p style="background-color:#efffff;color:#153944;font-size:50px;border-radius:2%;">Vogue Harmony</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:darkblue;font-size:16px;">Your Style, Perfected: Unleash the Power of Compatibility</p>', unsafe_allow_html=True)



cwdir=os.getcwd()

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

_success_text_upload = "File was uploaded successfully!"




"""
------------------------------------------------------------------------------------
"""





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
_category_list = [ 'bag', 'shoes', 'top', 'outwear', 'pants', 'eyewear', 'dress',  'skirt', 'watches', 'hats',
                   'earrings', 'rings',  'bracelet', 'necklace']

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

    df_category_all[i] = df_category
    i=i+1





# Catalog

def show_catalog(category_input,_label):

    df_temp_ = df_category_all[category_input].copy(deep=True)
    df_temp_['image_path'] = df_temp_['path'] + '/' + df_temp_['image']
    _img_file_ = list(df_temp_['image_path'].unique())

    img = image_select(
        label=_label,
        images=_img_file_,
        use_container_width=False,

    )
    # st.image(img)
    return img






tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(_category_list[:6])

with tab1:
    img1 = show_catalog(0, "Choose a bag from existing catalog")
    st.session_state['catalog_image1'] = img1

with tab2:
    img2 = show_catalog(1, "Choose a shoe from existing catalog")
    st.session_state['catalog_image2'] = img2

with tab3:
    img3 = show_catalog(2, "Choose a top from existing catalog")
    st.session_state['catalog_image3'] = img3

with tab4:
    img4 = show_catalog(3, "Choose an outer wear from existing catalog")
    st.session_state['catalog_image4'] = img4

with tab5:
    img5 = show_catalog(4, "Choose a pants from existing catalog")
    st.session_state['catalog_image5'] = img5

with tab6:
    img6 = show_catalog(5, "Choose an eye wear from existing catalog")
    st.session_state['catalog_image6'] = img6

# Option to Upload Images

_success_text_upload = "File was uploaded successfully!"


