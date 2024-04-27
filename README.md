# Firearm-detection-and-classification
**Main idea**

The project aims to address contemporary security challenges by harnessing cutting-edge technology to promptly identify firearms and classify them, thus empowering law enforcement agencies with enhanced threat assessment and response capabilities.

**Methodology**

![bnewee](https://github.com/Sejal-Dubey/Firearm-detection-and-classification/assets/140956763/7e62aec4-617d-4232-a721-e5a73d043563)

**Dataset description**
https://www.sciencedirect.com/science/article/pii/S2352340924000040#cebibl1

The dataset  comprised of 398 videos, each featuring an individual engaged in specific video surveillance actions. The ground truth for this dataset was expertly curated and is presented in JSON format (standard COCO), offering vital information about the dataset, video frames, and annotations, including precise bounding boxes outlining detected objects. The dataset encompasses three distinct categories for object detection: "Handgun", "Machine_Gun", and "No_Gun", dependent on the video's content.

**Models used**

**1.Harr-cascade classifier(for detection)**

The Haar cascade classifier is a machine learning-based approach used for object detection in images or video streams.It is computationally lightweight and offers fast inference times, making it ideal for real-time object detection tasks as it is specifically trained on gun datasets which is reason it is selected for this project.

**2.Deep learning models**

**a.Resnet 50 (for feature extraction)**

ResNet-50 is a deep convolutional neural network architecture composed of multiple layers and performs feature extraction by leveraging its deep convolutional layers to capture hierarchical features from input images, and the extracted features are subsequently used for firearm detection and classification tasks.

-Upon image uploading the detected image, preprocessing steps are undertaken to align with ResNet-50's requirements, involving resizing the image to 224x224 pixels and normalization for pixel value consistency.

-The preprocessed image proceeds through ResNet-50's convolutional layers, extracting hierarchical features at varying abstraction levels. Each layer refines features, capturing nuances from edges to high-level semantics. 

-The final two layers, responsible for classification, are omitted, isolating the feature extraction aspect. Extracted features undergo global average pooling to yield fixed-length feature vectors. These vectors, retrieved as numpy arrays, offer grounds for further processing or firearm classification endeavors by saving these features in a csv.


**b.Resnet 50 (for classification)**

The saved features.csv is used for training and testing by Resnet50.
The code trains a binary classification model using a dataset containing features extracted from firearm images. It splits the dataset into training and testing sets, defines a neural network model with dense layers using TensorFlow's Keras API, compiles the model with the Adam optimizer and binary crossentropy loss function, and then trains the model on the training data for a specified number of epochs and batch size. Finally, it evaluates the model's performance on the test data and test accuracy was found to be 93%.
The model was dumped to resnet50.h5 and it performed well by classifying classes correctly.

**3.Machine learning models(for classification)**

The saved features.csv is used for training and testing by these machine learning models:

**a.KNN**: KNN is a basic machine learning method where a new data point is classified or predicted by looking at the class or value of its closest neighbors from the training data.

**b.Random Forest**: Random Forest is an ensemble method that uses multiple decision trees. It reduces overfitting by training each tree on a random subset of the data and features.

**c.Adaboost**: AdaBoost is an ensemble learning technique that combines weak learners into a strong classifier. It sequentially trains models, focusing more on instances that were misclassified in previous rounds, to improve overall performance.

**d.Naives Bayes**: Naive Bayes is a straightforward probabilistic classifier that assumes features are independent. It's commonly used for text classification and other tasks, often performing well despite its simplicity.

Out of all these machine learning methods,KNN and Random forest worked better on the dataset with higher accuracy as compared to Naives Bayes and Adaboost,finally KNN and Random forest pickle files were used during deployment.

**Conclusion**:

-It was observed the machine learning models were not able to generalize well,after a certain point their limit execeeded and they started learning instead of classifying ,thus the model concluded to one class only.

-Resnet 50 model performed very well on unseen data which can further be expanded for classifying more variety of guns other than handgun and machinegun.

-The project was deployed on streamlit and further expansion include using yolov5 or other deep learning models for object detection and then exploring other deep learning models for classification. 

# Instructions on how to run the code

1.Run the harr-cascade classifier.ipynb ensuring that you have correctly provided the path of downloaded cascade.xml file and the folder where the detected region of interest will be saved.

2.Place a image in front of camera after running the code and the probable handgun and machine gun will be detected and saved as roi in the provided folder.

3.Run the st.py which is the streamlit code of deployment and local host of streamlit will be deployed on screen ,where give the path of any one suitable foi from saved folder and then choose the models from option and click predict button to get the predicted class of input image.

# Note:

**Link of all the pkl files ,h5 file and features.csv named augmented_combined_cropped_roi_handgun&machinegun.csv) is uploaded on mega due to size issue in github.**

csv- https://mega.nz/file/9ulGjKaB#Nt8oIM3m8xDUnfWmOlQMTWjoe-je6nFf2wvUC1fG4O8

resnet50.h5- https://mega.nz/file/4rEynb4T#XUKiRSA5unizlW1vfK5poFds2yYjSFpt-9J9uMCKq8E

knn.pkl- https://mega.nz/file/Ry0gHKjL#1rkLsfNSSj0BZvNwIpgA_ZzysgVAr5K0vaPDdJvf_cU

random forest-https://mega.nz/file/h2sA0RxI#cPHrXfrfT4vtD2jckHLZFV3hyQngExx6bWcsIUjhaTw



