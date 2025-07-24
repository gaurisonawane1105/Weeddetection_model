import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Function to set a custom background and font style
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)), 
                        url("https://images.pexels.com/photos/6216870/pexels-photo-6216870.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260");
            background-size: cover;
            background-position: center;
        }
        
    
   
      .container {
            background: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 800px;
            max-height: 90%
            margin: 0 auto;
            text-align: center;
        }
   
   
 

       
        """,
        unsafe_allow_html=True
    )

# Apply the custom background and font style
set_background()

# App title and header
import streamlit as st
# Custom CSS to underline the title
st.markdown("""
    <h1 style='text-decoration: underline;'>Image Classification Model</h1>
    """, unsafe_allow_html=True)

st.header('Weed Detection Tool')

# Load the trained model
model = load_model('D:\Mini Project\Dataset\WeedData\Image_classification.keras')

# Define the weed categories
data_cat = [
    'Argemone_mexicana_L',
    'Aristolochia_clematitis',
    'Asthma_Weed',
    'Clear_Soil',
    'Commelina_Diffusa',
    'Santa_Maria_weed',
    'Small_weed',
    'Wild_Lettuce'
]

# Input for image name
img_height, img_width = 700, 700
image = st.text_input(' Enter the image name', 'test2.jpeg')

try:
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make predictions using the model
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    # Display the uploaded image and results
    st.image(image, width=200, caption="Uploaded Image")
    st.write(' It is an image of ' + data_cat[np.argmax(score)] + '')
    st.write(' With accuracy of {:.2f}%'.format(np.max(score) * 100))
   
    

    

except Exception as e:
    st.error("🚫 Unable to load the image. Please check the file path and try again.")

# Footer with credits
st.markdown("""
    <footer>
        Powered by TensorFlow & Streamlit | Developed with ❤️
    </footer>
""", unsafe_allow_html=True)

