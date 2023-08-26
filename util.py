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
    # Resize the input image to (164, 164) using Lanczos resampling
    image = ImageOps.fit(image, (164, 164), Image.Resampling.LANCZOS)

    # Convert the resized image to a NumPy array
    image_array = np.asarray(image)

    # Normalize image pixel values to fall within the range [-1, 1]
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Set model input
    data = np.ndarray(shape=(1, 164, 164, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)

    # Determine the predicted class index
    index = np.argmax(prediction)

    # Get the predicted class name based on class_names list
    class_name = class_names[index]

    # Get the confidence score for the predicted class
    confidence_score = prediction[0][index]

    return class_name, confidence_score
