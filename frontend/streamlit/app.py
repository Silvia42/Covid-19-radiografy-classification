import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import json
import requests
# from keras.models import load_model
# from keras.preprocessing import image
import numpy as np
import io

# Emoji website https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title = "Covid-19 Detection with Radiography", page_icon = ':pill:', layout = 'wide')

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS template
local_css("style/style.css")

# Load Assets
lottie_coding = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_v6njxply.json")
inc_img = Image.open('../../graphs/Incheatmap1.png')
res_img = Image.open('../../graphs/Resheatmap2.png')
res25_img = Image.open('../../graphs/Res25heatmap3.png')

# Sidebar for Organization
st.title("COVID-19 Detection with Radiography")
section = st.sidebar.selectbox("Section", ['About Us', 'Live Model', 'Findings'])


# Load in page contents based on selection
if section == 'About Us':
    st.subheader("Welcome to Team Scooby's Transfer Learning Project")
    st.write("By: Kevin, Natalie, Silvia")
    st.write("[GitHub Link](https://github.com/DSML-Scooby/Covid-Radiography-Classification)")

    # Body section
    with st.container():
        st.write('----')
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("Project Description")
            st.write("##")
            st.write(
                """
                This project aims to help doctors more quickly and easily differentiate and diagnose 
                Covid-19 lung infections based on chest x-rays. We applied transfer learning with the use of
                pre-trained models such as InceptionV3 and ResNet50 to perform a classification of Covid vs. Normal
                labels. 
                """
            )
        
        with right_column:
            st_lottie(lottie_coding, height=300, key='xray')

elif section == 'Live Model':
    file = st.file_uploader('Upload Chest X-Ray Image Here', type = ['png', 'jpg'])
    if file is not None:
        st.text("Uploaded Image")
        img = Image.open(io.BytesIO(file))
        img = img.convert('RGB')
        target_size = (299,299)
        img = img.resize(target_size, Image.NEAREST)
        img = Image.img_to_array(img)




        # image = Image.open(file)
        # st.image(image, caption='Uploaded Image', use_column_width=True)
        # st.write("")
        # st.write("Classifying")
        # label = 

    # pass

    # # way to check if a button is clicked or not
    # if st.button("Get Prediction"):
    #     # Connect to Flask Predict
    #     url     =  'http://127.0.0.1:5000/.com:5000/predict'
    #     payload = json.dumps(inputs.values.tolist())
    #     resp    = requests.post(url, json = {'arr': payload})
    #     try:    
    #         st.write(np.array(resp.json()))
    #     except Exception as e:
    #         st.text(f"Could not process request because: {e}")
else:
    st.subheader("Findings on Models Tried")


    # Findings

    # First row
    with st.container():
        st.write("----")
        st.header("Models Tried")
        st.write("##")
        image_column, text_column = st.columns((1,2))
        with image_column:
            st.image(inc_img)
            # insert image
        with text_column:
            st.subheader("Base InceptionV3 Model")

    # Second row
    with st.container():
        # st.write("----")
        # st.header("Base ResNet Model")
        # st.write("##")
        image_column, text_column = st.columns((1,2))
        with image_column:
            st.image(res_img)
            # insert image
        with text_column:
            st.subheader("Base ResNet Model")

    # Third row
    with st.container():
        # st.write("----")
        # st.header("ResNet Model with Dropout Changed")
        # st.write("##")
        image_column, text_column = st.columns((1,2))
        with image_column:
            st.image(res25_img)
            # insert image
        with text_column:
            st.subheader("ResNet Model with Dropout Changed")

    



















# # Contact
# with st.container():
#     st.write("----")
#     st.header("Contact Form")
#     st.write('##')

#     # https://formsubmit.co/
#     # have to replace naked email address with string
#     contact_form = """
#     <form action="{email link}} " method="POST"> 
#         <input type="hidden" name="_captcha" value="false">
#         <input type="text" name="name" placeholder = "Your name" required>
#         <input type="email" name="email" placeholder = "Your email"required>
#         <textarea name="message" placeholder="Your message here" required></textarea>
#         <button type="submit">Send</button>
#     </form>
#     """
#     left_column, right_column = st.columns(2)
#     with left_column:
#         st.markdown(contact_form, unsafe_allow_html=True)
#     with right_column:
#         st.empty()