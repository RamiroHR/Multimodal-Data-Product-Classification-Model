import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 

import cv2
import joblib

import FusionModel_withVGG_tools as fm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input



def project_description():
    text = "At the core of this project lies the challenge to craft a robust model capable \
            of classifying products into 27 distinct categories. This is achieved by \
            meticulously analyzing not only product images but also the accompanying textual \
            descriptions, merging the visual and semantic facets of product understanding \
            into a unified classification framework."
    return text

def project_applications():
    text = "Effectively categorizing products based on a multitude of data is pivotal within\
            the dynamic e-commerce domain. It plays a multifaceted role, from refining product \
            recommendations and enabling personalized exploration to assisting customers in \
            pinpointing precisely what aligns with their preferences."
    return text



def model_description():
    models = {}

    models['Null'] = "This models predicts always the most common class, \
        which for the Rakuten catalogue is the 'piscine_spa' category (code 2583):"

    models['Text_Models'] = "These models only use the information provided on the **Title** \
        and the **Description** fields of the product."
    
    models['Image_Models'] = 'These models only use the information provided on the **Image** \
         of the product.'

    models['Fusion'] = "The Fusion Model is an amalgamation of four deep learning models, \
        with three of them trained on the Rakuten product catalog and one acquired through transfer \
        learning. These models are meticulously engineered to intricately interconnect, \
        resulting in the creation of a significantly more powerful and efficient classification model."
    
    models['hl_image'] = "This model is crafted from the backbone of a pre-trained VGG-16 model, which includes \
        13 convolutional layers, serving as an effective image processing component for extracting visual features. \
        Following this, a multilayer perceptron with two hidden layers is incorporated to learn from the VGG-16's output. \
        The headless model is finalized by removing the last classification layer of the mulltilayer perceptron after being trained."
    
    models['hl_text'] = "Constructed around a multilayer perceptron with a single hidden layer containing 256 units, \
        the headless text model excels in text-based feature extraction. After training, the last classification layer is removed."

    models['fusion_head'] = "The head of the fusion model derives insights from the output vectors extracted by the preceding headless models. \
        These vectors represent highly relevant text and image features extracted from our dataset. The head model comprises a multilayer \
        perceptron with two dense hidden layers, trained on the concatenated and newly extracted text and image features."

    return models


def performance_description(key):
    text = {}

    text['perf_comparison'] = "The bar plot below shows a comparison of the performance between \
        the constitutent **text model** (**NN**) and **image model** (**VGG+NN**) with the final **Fusion model** \
        (**FM**). The fusion model performs much better than the component models, achieving a test \
        accuracy of **81.9 %**. Regarding the computation time, the Fusion model trains in about 13 minutes \
        and remains relatively fast for computing predictions, taking 2.7 seconds to predict the category of 16984 \
        products (test set) running a CPU."
    
    text['confusion_matrix'] = "The diagram below presents the confusion matrix for the Fusion model, which has been \
        assessed against the test dataset (17 K products). This matrix \
        serves as a comprehensive summary of the model's performance across diverse classes, \
        highlighting the alignment between its predictions and the actual classes. In an \
        effective multi-class classification model, an optimal confusion matrix showcases a \
        robust diagonal line with minimal scattered non-diagonal elements. This pattern reflects \
        the model's consistent and accurate predictions for each class within the classification task."
    
    text['classification_report'] = "The figure below presents the classification report of \
        the Fusion model, offering class-specific metrics like precision, recall, and F1-score. \
        This report enables a detailed assessment of how well the model predicts each class. It \
        serves as a valuable tool in pinpointing the model's strengths and weaknesses, detecting \
        potential class imbalances, and quantifying its overall performance. \
        In the following table, the lighter the cell's color, the better the performance."

    return text[key]



def get_text_model_comparison(chosen_model):
    imagefile = './images/text_models_comparison'
    
    text_models_dict = {'Random Forest': imagefile + '_RF.png',
                        'Support Vector Machines': imagefile + '_SVC.png',
                        'Dense Neural Network': imagefile + '_DNN.png'}
    
    return text_models_dict[chosen_model]


def demo_instructions():
    text = """
            You can try the fusion model by yourself with real data. Here it is how:  
            Go to the Rakuten France website [link](%s) and chose a product, then:  
            > - Copy the product title and paste it into the 'Title' field below.  
            > - Copy the product title and paste it into the 'Description' field below.  
            > - Download the product image and upload it into the 'Image' field below.   
            """
    return text



def get_predictions(sample_title, sample_description, sample_image):
    sample_pred_code, sample_pred_class = 'unknown', 'unknown' 

    sample_image_hl_output = get_image_hl_model_output(sample_image)

    sample_text_hl_output = get_text_hl_model_output(sample_title, sample_description)

    ## past argument in the right order to concatenate text + img
    sample_fusion_data = get_fusion_data(sample_text_hl_output, sample_image_hl_output)

    fusion_model = get_trained_model('fusion')

    ## run headless model
    sample_pred_vector = fusion_model.predict(sample_fusion_data)

    ## translate model prediction into a meaningfull output

    # sample_pred_code, sample_pred_class = get_decoded_predictions(sample_pred_vector)
    sample_pred_code, sample_pred_class, pred_confidences = get_decoded_predictions_with_confidence(sample_pred_vector, num = 3)
    return sample_pred_code, sample_pred_class, pred_confidences

 

def get_decoded_predictions(sample_pred_vector):

    path = './trained_models/'
    target_encoder = joblib.load(path + "2308281220_text_target_encoder")

    ## reverse one hot encoding
    sample_pred_label = sample_pred_vector.argmax(axis = 1)

    ## revert label encoder, get predcited code
    sample_pred_code = target_encoder.inverse_transform(sample_pred_label)[0]
    
    ## get predicted class name
    path = '../'
    product_class = pd.read_csv(path + 'product_class.csv', sep = ';')
    product_class.drop('target', axis = 1, inplace = True)
    
    sample_pred_class = product_class[product_class['prdtypecode']==sample_pred_code]['prodtype'].values
    
    return sample_pred_code, sample_pred_class


def get_decoded_predictions_with_confidence(sample_pred_vector, num = 3):

    path = './trained_models/'
    target_encoder = joblib.load(path + "2308281220_text_target_encoder")

    ## reverse one hot encoding
    pred_labels = sample_pred_vector.argsort(axis = 1)[:,-3:][:,::-1].T

    confidences = [sample_pred_vector[0,label][0] for label in pred_labels]

    ## revert label encoder, get predcited code
    pred_codes = [target_encoder.inverse_transform(label)[0] for label in pred_labels]
    
    ## get predicted class name
    path = '../'
    product_class = pd.read_csv(path + 'product_class.csv', sep = ';')
    product_class.drop('target', axis = 1, inplace = True)
    
    pred_classes = [ product_class[product_class['prdtypecode']==code]['prodtype'].values[0] for code in pred_codes]
    
    return pred_codes, pred_classes, confidences


def get_fusion_data(sample_text_hl_output, sample_image_hl_output):

    path = './trained_models/'
    text_hl_output_scaler = joblib.load(path + '2309012059_text_hl_data_output_scaler')
    image_hl_output_scaler = joblib.load(path + '2309012059_image_hl_data_output_scaler')

    sample_fusion_data = fm.get_sample_fusion_data(sample_text_hl_output, sample_image_hl_output, 
                                               text_hl_output_scaler, image_hl_output_scaler)
    
    return sample_fusion_data



def get_text_hl_model_output(sample_title, sample_description):

    ## wrap text data into a dataframe
    sample_text = wrap_text_input(sample_title, sample_description)

    ## preprocess text data: cleaning, feature engineering, etc
    sample_text_preprocessed = fm.preprocess_text_data(sample_text, verbose = False)

    ## transform text data: vectorization, etc
    path = './trained_models/'
    token_len_scaler = joblib.load(path + '2308281220_token_len_scaler')
    language_encoder = joblib.load(path + '2308281220_language_encoder')
    lemmas_vectorizer = joblib.load(path + '2308281220_lemmas_vectorizer')

    sample_text_transformed = fm.transform_sample_text(sample_text_preprocessed, token_len_scaler,
                                                   language_encoder, lemmas_vectorizer, verbose = 1)

    ## get text headless model
    model = get_trained_model('hl_txt_model')

    ## run headless model
    sample_text_hl_output = model.predict(sample_text_transformed)

    return sample_text_hl_output


def wrap_text_input(title, description):

    text_dict = {'title' : [title],
                'description' : [description]}

    text_df = pd.DataFrame(text_dict, columns = text_dict.keys())

    return text_df



def get_image_hl_model_output(sample_image):
    
    ## Preprocess and transform data
    sample_image_preprocessed = preprocess_sample_image(sample_image)

    ## Load model
    model = get_trained_model('hl_img_model')
    # model.summary(print_fn=lambda x: st.text(x))

    ## apply hl_image_model
    sample_image_hl_output = model.predict(sample_image_preprocessed)

    return sample_image_hl_output



def get_trained_model(model_key):

    ## define dictionary with keys
    trained_models = { 
        'RF_txt' : '2308181133_text_rf_trained.joblib',
        'hl_img_model' : '2309012056_image_headless_model_pack.keras',
        'hl_txt_model' : '2309012056_headless_text_model.keras',
        'fusion' : '2309012136_fusion_model_trained.keras'}
    
    path = './trained_models/'
    model_file = path + trained_models[model_key]

    ## get model with joblib
    if model_key in ['RF_txt']:
        model = joblib.load(model_file)

    elif model_key in ['hl_img_model','hl_txt_model','fusion']:
        model = load_model(model_file)
     
    return model



def preprocess_sample_image(sample_image):

    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(sample_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    sample_image_cropped = fm.crop_image(opencv_image, threshold = 230)
    sample_image_resized = cv2.resize(sample_image_cropped, (224, 224))

    ## Preprocess af for VGG feature extractor: The images are converted from RGB to BGR, 
    ## then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.

    sample_image_batch = np.expand_dims(sample_image_resized, axis=0)
    sample_image_scaled = preprocess_input(sample_image_batch) / 255

    return sample_image_scaled


