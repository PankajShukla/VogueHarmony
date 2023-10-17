!pip install streamlit-image-select
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
    page_title="Home",
    layout="wide"
)

st.markdown(f'<p style="background-color:#efffff;color:#153944;font-size:50px;border-radius:2%;">Vogue Harmony</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:darkblue;font-size:16px;">Your Style, Perfected: Unleash the Power of Compatibility</p>', unsafe_allow_html=True)



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

col1.markdown("""
<html><body>
<iframe src={gif} width="400" height="300" frameBorder="0" class="giphy-embed" allowFullScreen>
</iframe>
</body></html>
""".format(gif=gif_list[0]), unsafe_allow_html=True)


col2.markdown("""
<html><body>
<iframe src={gif} width="400" height="300" frameBorder="0" class="giphy-embed" allowFullScreen>
</iframe>
</body></html>
""".format(gif=gif_list[1]), unsafe_allow_html=True)

col3.markdown("""
<html><body>
<iframe src={gif} width="400" height="300" frameBorder="0" class="giphy-embed" allowFullScreen>
</iframe>
</body></html>
""".format(gif=gif_list[2]), unsafe_allow_html=True)




"""

------------------------------------------------------------------------------------
"""


st.markdown(f'<center><p style="background-color:#ffffef;color:darkblue;font-size:20px;"><b> 3 Simple steps to get the best outfit for yourself </b>  </p></center>',
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