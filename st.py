import streamlit as st
import joblib
import numpy as np
from PIL import Image  # for image processing
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import functional as F
import pickle
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout

# Define a custom feature extraction model by removing the final classification layer
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

# Initialize the custom feature extraction model
feature_extractor = FeatureExtractor()
feature_extractor.eval()

# Function to extract features from a cropped ROI using ResNet-50
def extract_features_from_roi(roi):
    # Preprocess the ROI
    roi = F.resize(roi, (224, 224))  # Resize to fit ResNet-50 input size
    roi = F.to_tensor(roi)  # Convert to tensor
    roi = F.normalize(roi, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize

    # Add batch dimension
    roi = roi.unsqueeze(0)

    # Extract features using ResNet-50
    with torch.no_grad():
        features = feature_extractor(roi)

    return features.squeeze().numpy()  # Remove batch dimension and return features as numpy array

# Function to predict class using ResNet50 model
def predict_class_resnet(image_path):
    # Load the pre-trained ResNet50 model
    resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Add a Dense layer to ResNet50 to match the expected input shape of the classifier model
    output_layer = Dense(2049, activation='relu')(resnet_model.output)
    output_layer = Dropout(0.5)(output_layer)

    # Load your saved classifier model
    classifier_model = load_model(r'C:\Users\Sejal\Downloads\yolo_source\freshresnet50model.h5')

    # Combine the ResNet50 base model with the classifier model
    combined_model = tf.keras.Model(inputs=resnet_model.input, outputs=classifier_model(output_layer))

    # Load and preprocess the input image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to match ResNet50 input size
    img_array = np.expand_dims(np.array(img), axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image

    # Use the combined model to predict the class
    prediction = combined_model.predict(img_array)

    # Get the predicted class label
    if prediction[0][0] < 0.5:
        predicted_class = 'handgun'
    else:
        predicted_class = 'machinegun'

    return prediction, predicted_class


def main():
    st.title("Firearm Detection and Tracking")

    # Function to load the pickled model
    def load_model(model_path):
        model = joblib.load(model_path)
        return model

    # Load the model outside the main function
    random_forest_model_path = r"C:\Users\Sejal\Downloads\yolo_source\random_forest_model.pkl"
    random_forest_model = load_model(random_forest_model_path)

    knn_model_path = r'C:\Users\Sejal\Downloads\yolo_source\new_knn_classifier_final.pkl'
    knn_model = load_model(knn_model_path)

    st.write(
        """ Our project aims to develop a firearm detection and tracking system using computer vision techniques. 
        With the increasing concern about firearm-related incidents, particularly in security and surveillance contexts, 
        an automated system capable of accurately detecting and tracking firearms in images and videos can provide 
        valuable support to law enforcement and security personnel. Our goal is to create a robust and efficient solution 
        that can detect firearms in various environments and lighting conditions, track their movements in real-time, 
        and provide visual feedback to aid in situational awareness and decision-making."""
    )

    st.write("""
    This Streamlit app demonstrates firearm detection and tracking using computer vision techniques.
    
    *Features*:
    - Detects firearms in images or video streams.
    - Tracks the detected firearms in uploaded videos.
    - Provides visualization of the detection and tracking results.
    
    *How to Use*:
    - Upload an image or video containing firearms.
    - The app will detect the firearms and display bounding boxes around them.
    - If it's a video, you can choose to track the firearms in real-time.
    - Explore the detection and tracking results!
    """)
    Explore_Dataset = st.sidebar.selectbox("Explore Dataset",["Hand Gun","Machine Gun","Annotated Frames"])

    if Explore_Dataset == "Hand Gun":
        st.write("*Hand Gun Images:*")

        # Define the width of each image column
        col_width = 200

        # Display the images in two columns
        col1, col2 = st.columns(2)

        # Image paths
        image_paths = [
            r"C:\Users\Sejal\Downloads\yolo_source\frame62.jpg",
            r"C:\Users\Sejal\Downloads\yolo_source\frame75.jpg",
        ]

        # Display images in the first column
        with col1:
            st.image(image_paths[0], caption="Image 1", width=col_width)
            st.image(image_paths[1], caption="Image 2", width=col_width)

    if Explore_Dataset == "Machine Gun":
        st.write("*Machine Gun Images:*")

        # Define the width of each image column
        col_width = 200

        # Display the images in two columns
        col1, col2 = st.columns(2)

        # Image paths
        image_paths = [
            r"C:\Users\Sejal\Downloads\yolo_source\frame94.jpg",
            r"C:\Users\Sejal\Downloads\yolo_source\frame120.jpg",
        ]

        # Display images in the first column
        with col1:
            st.image(image_paths[0], caption="Image 1", width=col_width)
            st.image(image_paths[1], caption="Image 2", width=col_width)

    if Explore_Dataset == "Annotated Frames":
        st.write("*Annotated Images:*")

        # Define the width of each image column
        col_width = 200

        # Display the images in two columns
        col1, col2 = st.columns(2)

        # Image paths
        image_paths = [
            r"C:\Users\Sejal\Downloads\yolo_source\annotated_frame67.jpg",
            r"C:\Users\Sejal\Downloads\yolo_source\annotated_frame158.jpg",
            r"C:\Users\Sejal\Downloads\yolo_source\annotated_frame147.jpg",
            r"C:\Users\Sejal\Downloads\yolo_source\annotated_frame90.jpg",
        ]

        # Display images in the first column
        with col1:
            st.image(image_paths[0], caption="Image 1", width=col_width)
            st.image(image_paths[1], caption="Image 2", width=col_width)

        # Display images in the second column
        with col2:
            st.image(image_paths[2], caption="Image 3", width=col_width)
            st.image(image_paths[3], caption="Image 4", width=col_width)


    # Dictionary to store model descriptions
    model_descriptions = {
        "Random Forest": "Random Forest is an ensemble learning method used for classification and regression tasks in machine learning.",
        "kNN": "k-Nearest Neighbors (kNN) is a simple instance-based learning algorithm used for both classification and regression tasks in machine learning. ",
        "Resnet50":"ResNet-50 is a convolutional neural network architecture that is widely used for various computer vision tasks, including image classification, object detection, and image segmentation"
    }

    # Selectbox for choosing the model in the sidebar
    model_used = st.sidebar.selectbox("Models Used", ["Resnet50","Random Forest", "kNN"])

    if model_used == "Resnet50":
        st.write("*Resnet50 Training snapshot:*")
        st.image(r"C:\Users\Sejal\Downloads\yolo_source\Screenshot 2024-04-24 131112.png", caption="Resnet50")
        # You can display metrics here if needed

    if model_used == "Random Forest":
        st.write("*Random Forest Metrics:*")
        st.image(r"C:\Users\Sejal\Downloads\yolo_source\randomforestreport.png", caption="Classification_Report")
        st.write("*Random Forest Confusion-matrix:*")
        st.image(r"C:\Users\Sejal\Downloads\yolo_source\confusionmatrixrf.png",caption="confusion-matrix")
        st.write("*Best Parameters*:")
        st.image(r"C:\Users\Sejal\Downloads\yolo_source\gridsearchcv.png",caption="grid_search_cv")
        # You can display metrics here if needed

    if model_used == "kNN":
        st.write("*kNN Metrics:*")
        st.image(r"C:\Users\Sejal\Downloads\yolo_source\knnreport.png", caption="Classification_Report")
        st.write("*kNN Confusion-matrix:*")
        st.image(r"C:\Users\Sejal\Downloads\yolo_source\confusionmatrixknn.png",caption="confusion-matrix")
        # You can display metrics here if needed

    # Display the selected model description in the sidebar
    if model_used in model_descriptions:
        st.sidebar.write(f"*Description of {model_used}:*")
        st.sidebar.write(model_descriptions[model_used])

    Upload_files = st.sidebar.selectbox("Upload Files", ["Upload Image"])
    height = 300
    width = 300
    flattened_image = None  # Initialize flattened_image variable

    # Upload and display image if "Upload Image" is selected
    if Upload_files == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image")
            prediction_method = st.selectbox("Prediction Method", ["Random Forest", "kNN", "ResNet50"])
            if st.button("Predict"):  # Add a predict button
                if prediction_method == "ResNet50":
                    # Perform prediction using ResNet50 model
                    prediction_values, predicted_class = predict_class_resnet(uploaded_image)
                    st.write("Prediction values:", prediction_values)
                    st.write("Predicted class:", predicted_class)
                else:
                    image = Image.open(uploaded_image)  # Open the uploaded image
                    # Extract features from the uploaded image
                    input_features = extract_features_from_roi(image)

                    # Reshape the input features to have a single sample
                    input_features = input_features.reshape(1, -1)

                    # Initialize PCA for dimensionality reduction
                    pca = PCA(n_components=1)
                    # Fit PCA model to input features and transform them
                    input_features_pca = pca.fit_transform(input_features)

                    if prediction_method == "Random Forest":
                        # Perform prediction using Random Forest model
                        prediction = random_forest_model.predict(input_features_pca)
                    elif prediction_method == "kNN":
                        # Perform prediction using k-NN model
                        prediction = knn_model.predict(input_features)
                    st.write("Prediction:", prediction)

    st.write("""
    *Team Members:*
    - Drishtti Narwal
    - Sejal Dubey 
    - Hemang Sharma 
    """)

if __name__ == "__main__":
    main()
