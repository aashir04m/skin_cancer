import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import cv2


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (164, 164)
    # image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    # print("image", image)

    # # convert image to numpy array
    # image_array = np.asarray(image)

    # # normalize image
    # normalized_image_array = (image_array.astype(np.float32) / 255.0) - 1

    image = cv2.resize(image, (28, 28))
    
    prediction = model.predict(image.reshape(1, 28, 28, 3))
                                             
    max_prob = max(result[0])

    class_ind = list(result[0]).index(max_prob)
    print ("class ind",class_ind)
    class_name = classes[class_ind]                                         

    # set model input
    # data = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)
    # data[0] = normalized_image_array

    # # make prediction
    # prediction = model.predict(data)
    # print("prediction", prediction)
    # index = np.argmax(prediction)
    # print("index", index)
    # class_name = 0 if prediction[0][0] > 0.95 else 1

    # class_name = "PNEUMONIA" if class_name == 0 else "NORMAL"

    # if class_name == 0:
    #     class_name = "PNEUMONIA"
    # else:
    #     class_name = "NORMAL"
    # class_name = class_names[index]
    confidence_score = prediction[0][0]

    return class_name, confidence_score
