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
    page_title="Complete Outfit",
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



img1 = st.session_state['catalog_image1']
img2 = st.session_state['catalog_image2']
img3 = st.session_state['catalog_image3']
img4 = st.session_state['catalog_image4']
img5 = st.session_state['catalog_image5']
img6 = st.session_state['catalog_image6']



# 7th Image
df_temp = df_predict.copy(deep=True)
df_temp['image_path'] = df_temp['path'] + '/' + df_temp['image']
_img_file = list(df_temp['image_path'].unique())
_img_file = random.sample(_img_file, len(_img_file))
_img_file = _img_file[0]





"""
------------------------------------------------------------------------------------
"""

text_upload = '<center><p style="background-color:#ffffef;color:darkblue; font-size: 18px;">We have Completed your outfit</p></center>'

st.markdown(text_upload, unsafe_allow_html=True)


fig_, axes_ = plt.subplots(1, 7, figsize=(20, 5))
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


axes_[6].imshow(mpimg.imread(_img_file))
axes_[6].set_title('outfit ' + str(7))
axes_[6].set_xticklabels([])
axes_[6].set_yticklabels([])

axes_[6].set_xticks([])
axes_[6].set_yticks([])

axes_[6].spines['bottom'].set_color('red')
axes_[6].spines['top'].set_color('red')
axes_[6].spines['left'].set_color('red')
axes_[6].spines['right'].set_color('red')



st.pyplot(fig_)



"""
------------------------------------------------------------------------------------
"""

