import time
import numpy as np
import pandas as pd
import os 
import math

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def date_time():
    '''
    get date and time in string format '_yymmdd_hhmm'
    at the moment the function is called.
    '''
    from datetime import date, datetime
    
    today = date.today()
    now = datetime.now() 

    return today.strftime("%Y%m%d")[2:] + now.strftime("%H%M")



    
def save(datasets, types, names,  path, doit = False, verbose = True):
    '''
    Save each dataframe in dataframes with the respective name in names.
    Save at the specified path. 
    '''
          
    if doit == True:

        splitting_time = date_time()

        for data, type_, name in zip(datasets, types, names):
            filename = splitting_time + '_' + name 
            
            if type_ == 'dataframe':     
                filename = filename + '.csv'
                data.to_csv(path + filename, header = True, index = True)  # need index after train_test_split
                print("Saved dataset: %s" % (path+filename)) if verbose else None

            elif type_ == 'array':
                filename = filename + '.npy'
                np.save(os.path.join(path, filename), data)
                print("Saved dataset: %s" % (path+filename) ) if verbose else None
                
            elif type_ == 'sparseMatrix':
                filename = filename + '.npz'
                from scipy import sparse
                sparse.save_npz(os.path.join(path, filename), data)
                print("Saved sparseMatrix : %s" % (path+filename) ) if verbose else None
#                 your_matrix_back = sparse.load_npz("yourmatrix.npz")
                
            elif type_ == 'transformer':
                filename = filename
                import joblib
                joblib.dump(data, os.path.join(path, filename))
                print("Saved transformer: %s" % (path+filename) ) if verbose else None
#                 my_scaler = joblib.load('scaler.gz')

            elif type_ == 'XLMatrix':
                filename = filename + '.npy'
                import joblib
                joblib.dump(data, os.path.join(path, filename))
                print("Saved large matrix: %s" % (path+filename) ) if verbose else None
#                 my_matrix = joblib.load('matrix')

            elif type_ == 'arrayXL':
                filename = filename + '.npz'
                np.savez_compressed( path + filename, array = data)
                print("Saved compressed large array: %s" % (path+filename) ) if verbose else None
#                 loaded_data = np.load(save_path)
#                 loaded_array = loaded_data['array']

        return
    
    else:
        print("Datasets were not saved locally. Set doit = True to store them") if verbose else None
        return




####################################################################################################################
    
    
        
def preprocess_text_data(dataframe, verbose = True):
    
    df = dataframe.copy()
    
    # rename variable 
    df.rename({'designation':'title'}, axis = 1, inplace = True)
    print("Column 'designation' has been renamed as 'title' \n")
    
    
    # Feature engineering: title_descr
    concatenate_variables(df, 'title', 'description', nans_to = '', separator =' \n ', \
                          concat_col_name = 'title_descr', drop = False, verbose = verbose)

    
    # HTML parse & lower case
    html_parsing(df, 'title_descr', verbose = verbose)
   

     # Tokenize and lemmatize
    import nltk
    nltk.download('wordnet', quiet = True)
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import WordNetLemmatizer
    
    tokenizer = RegexpTokenizer(r'\w{3,}')
    lemmatizer = WordNetLemmatizer()

    get_lemmatized_tokens(df, 'title_descr', tokenizer, 'lemma_tokens', lemmatizer, uniques = True)
    
    
    ## Get language
    get_language(df, 'title_descr', correct = True, get_probs = False, verbose = verbose)
    
    
    ## Remove stop words according to language
    remove_stop_words(df, 'lemma_tokens', 'lemma_tokens', 'language', verbose = verbose)
    
    
    ## feature engineering token_length
    get_token_length(df, 'lemma_tokens', 'text_token_len', verbose = verbose)
    
    
    return df




def concatenate_variables(df, col1, col2, nans_to, separator, concat_col_name, drop = False, verbose = True):
    '''
    Replace NaNs in col1 and col2 with string nans_to
    Concatenate col1 and col2 using a separator string separator
    Drop columns if specified by to drop (list of columns)
    save in a new variable named by concat_col_name
    '''
    
    ## Replace NaN's in description with empty string
    df[col1] = df[col1].fillna(nans_to)
    df[col2] = df[col2].fillna(nans_to)

    ## Concatenate col1 with col2
    df[concat_col_name] = df[col1] + separator + df[col2]
   
    if verbose:
        print("Columns '%s' and '%s' have been concatenated in a new variable '%s' \n" %(col1,col2,concat_col_name))
    
    
    ## drop columns
    if drop:
        df.drop(col1, axis = 1, inplace = True)
        df.drop(col2, axis = 1, inplace = True)

        if verbose:
            print("%s and %s have been dropped" %(col1,col2))





def html_parsing(df, col_to_parse, verbose = False):
    '''
    HTML parse and lower case text content in col_to_parse
    '''
    from bs4 import BeautifulSoup

    import warnings
    from bs4 import MarkupResemblesLocatorWarning #GuessedAtParserWarning
    warnings.filterwarnings('ignore', category = MarkupResemblesLocatorWarning)
    
    
    t0 = time.time()
    
#     with warnings.catch_warnings():     ## disable warnings when BeautifulSoup encounters just plain text.
#         warnings.simplefilter("ignore")
#         warnings.filterwarnings('ignore', category = MarkupResemblesLocatorWarning)
        
    df[col_to_parse] = [BeautifulSoup(text, "lxml").get_text().lower() for text in df.loc[:,col_to_parse]]  #lxml, html.parser

    t1 = time.time()
    
    if verbose: 
        print(f"Column '{col_to_parse}' has been successfully HTML parsed and decapitalized.")
        print("\t HTML parsing takes %0.2f seconds \n" %(t1-t0))



def get_lemmatized_tokens(df, col_to_tokenize, tokenizer, tokenized_col, lemmatizer, uniques = False, verbose = True):
    '''
    For each row creates a list of tokens obtained from 'col_to_tokenize' column by tokenizing the text.
    Then lemmatize each word in the list, for each row.
    If unique = True, remove duplicated from each list of lemmas using set(). Keep the order of the words in list.
    Store list of lemmas in a new variable 'tokenized_col'
    '''    
    
    t0 = time.time()
    
    all_token_list = [tokenizer.tokenize(text) for text in df.loc[:,col_to_tokenize]]
    all_lemmatized_list = [ [lemmatizer.lemmatize(t) for t in token_list] for token_list in all_token_list ]

    if uniques :    
        df[tokenized_col] = [sorted( set(lemma_list), key=lemma_list.index ) for lemma_list in all_lemmatized_list ]
    else:    
        #df[tokenized_col] = [tokenizer.tokenize(text) for text in df.loc[:,col_to_tokenize]]
        df[tokenized_col] = all_lemmatized_list

    t1 = time.time()
    
    if verbose:
        print(f"Column '{col_to_tokenize}' has been successfully tokenized.")
        print("\t Tokenization + Lemmatization takes %0.2f seconds \n" %(t1-t0))

        
        

def get_language(df, text_col, correct = False, get_probs = False, verbose = True):
    
    from langid.langid import LanguageIdentifier, model
    from langid.langid import set_languages
    
    ## Main language identification
    ## instantiate identifier to get probabilities
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    
    ## identify language
    time_0 = time.time()

    languages_probas = [identifier.classify(text) for text in df[text_col]]
    
    time_1 = time.time()
    
    
    ## correct languages detected with low confidence
    if correct:
        ## restricted identifier to few languages
        identifier_2 = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        identifier_2.set_languages(langs=['fr','en'])
        
        for i, idx in zip( range(len(languages_probas)), df.index):    
            if languages_probas[i][1] < 0.9999:
                languages_probas[i] = identifier_2.classify(df.loc[idx,text_col]) ##### error
                
    time_2 = time.time()

    
    ## save detection in dataframe
    df['language'] = list(np.array(languages_probas)[:,0])
    
    if get_probs:
        df['lang_prob'] = [float(p) for p in np.array(languages_probas)[:,1]]
    
    
    if verbose:
        print("Main language detection takes %0.2f minutes." %((time_1 - time_0)/60) )
        if correct:
            print("\t Language detection correction takes %0.2f seconds \n" %(time_2 - time_1) )    
    return



def import_stop_words(language):
    '''
    Import list of stop words from the indicated language.
    If language is not in the list of the top 4 languages, use FR+EN by default.
    '''
    
    # top 4 languages in dataset
    available = ['fr', 'en', 'de', 'it']
    
    if language == 'fr':
        from spacy.lang.fr.stop_words import STOP_WORDS as stop_fr
        return list(stop_fr)
        
    elif language == 'en':
        from spacy.lang.en.stop_words import STOP_WORDS as stop_en
        return list(stop_en)

    elif language == 'de':
        from spacy.lang.de.stop_words import STOP_WORDS as stop_de
        return list(stop_de)

    elif language == 'it':
        from spacy.lang.it.stop_words import STOP_WORDS as stop_it
        return list(stop_it)

    else:
        from spacy.lang.fr.stop_words import STOP_WORDS as stop_fr
        from spacy.lang.en.stop_words import STOP_WORDS as stop_en
        return list(stop_fr) + list(stop_en)



    
def remove_stop_words(df, col_to_clean, col_result, col_language, verbose = False):
    '''
    Remove the stop words from each token list in df[col_to_clean] according to the detected language df['language']
    Store the cleaned token list in a new variable df[col_result]
    If col_result result does not exist in dataframe, intialize with empty strings (object dtype)
    '''

    t0 = time.time()
    
    new_name = col_result 
    if new_name not in df.columns:
        df[new_name] = ''  # initilize col as 'object' type
    
    for i, token_list, language in zip(df.index, df[col_to_clean], df[col_language]):
    
        stop_words = import_stop_words(language)        
        df.at[i, new_name] = [token for token in token_list if token not in stop_words]
    
    t1 = time.time()
        
    if verbose:
        print("Removing stop-words takes %0.2f seconds. \n" %(t1-t0))
    

    

def get_token_length(df, col_with_tokens, col_with_length, verbose = False):
    '''
    Creates a new variable measuring the number of tokens in column col_with_tokens
    '''
    t0 = time.time()
    
    df[col_with_length] = [len(token_list) for token_list in df[col_with_tokens] ]
    
    t1 = time.time()
    
    if verbose:
        print("Token counting takes %0.2f seconds. \n" %(t1-t0))


####################################################################################################################


def get_text_data(X_train, X_test, y_train, y_test):

    ## transform feature variables
    X_transformed_train, X_transformed_test, text_transformer = transform_features(X_train, X_test)
    
    ## transform target variables
    y_transformed_train, y_transformed_test, target_transformer = transform_target(y_train, y_test)
    
    ## pack datasets
    text_data = {'X_train' : X_transformed_train,
                 'X_test'  : X_transformed_test}
    
    targets = {'y_train' : y_transformed_train,
               'y_test'  : y_transformed_test}
    
    ## pack tranformers trained
#     text_transformers = {'X_transformer' : X_transformer,
#                          'y_transfomer'  : y_transformer}
    
    return text_data, targets, text_transformer, target_transformer
    

    
def transform_target(y_train, y_test):
    '''
    transform target varibale as needed for the choosen model.
    '''

    ## Label encoder
    from sklearn.preprocessing import LabelEncoder
    target_encoder = LabelEncoder()
    
    y_train_encoded = target_encoder.fit_transform(y_train.squeeze())
    y_test_encoded = target_encoder.transform(y_test.squeeze())
    

    ## One Hot encoder
    from tensorflow.keras.utils import to_categorical

    yy_train = to_categorical(y_train_encoded, dtype = 'int') 
    yy_test = to_categorical(y_test_encoded, dtype = 'int')   
    
    
    return yy_train, yy_test, target_encoder
    
    
        
def transform_features(X_train, X_test):
    '''
    Select features to keep.
    Transform data to Nd-array to feed the model
    Transform target varibale as well.
    '''
    
    # scale_text_token_len
    text_len_scaled_train, text_len_scaled_test, scaler = scale_feature(X_train, X_test, 'text_token_len')
    
    # One hot encode languages
    language_encoded_train, language_encoded_test, encoder = encode_feature(X_train, X_test, 'language')
    
    # Text vectorization
    text_vector_train, text_vector_test, vectorizer = vectorize_feature(X_train, X_test, 'lemma_tokens')
    
    # assembly all to return a single 2D array
    from scipy.sparse import hstack
    
    X_train_transformed = hstack(( text_len_scaled_train, language_encoded_train, text_vector_train ))
    X_test_transformed = hstack(( text_len_scaled_test, language_encoded_test, text_vector_test ))

    # transformers
    transformers = {'token_len_scaler' : scaler,
                   'language_encoder'  : encoder,
                   'lemmas_vectorizer' : vectorizer}
    
    
    return X_train_transformed, X_test_transformed, transformers


   
def scale_feature(X_train, X_test, col_to_scale):
    '''
    Scale feature using the specified scaler.
    '''
    
    from sklearn.preprocessing import MinMaxScaler 

    scaler = MinMaxScaler()

    col_scaled_train = scaler.fit_transform(X_train[[col_to_scale]])
    col_scaled_test = scaler.transform(X_test[[col_to_scale]])

    return col_scaled_train, col_scaled_test, scaler


def encode_feature(X_train, X_test, col_to_encode):
    '''
    One Hot encode categorical feature.
    Returns a matrix (sparse)
    '''
    
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(handle_unknown='ignore') #ignore, infrequent_if_exist

    col_encoded_train = encoder.fit_transform( X_train[[col_to_encode]] )
    col_encoded_test = encoder.transform( X_test[[col_to_encode]] )

    return col_encoded_train, col_encoded_test, encoder



def vectorize_feature(X_train, X_test, col_to_vectorize):
    '''
    vectorize text using custom tokenizer.
    retruns a sparece matrix.
    '''

    from sklearn.feature_extraction.text import TfidfVectorizer
    import warnings
    warnings.filterwarnings('ignore', category = UserWarning)

    vectorizer = TfidfVectorizer(tokenizer = do_nothing, lowercase=False, max_features=5000) #max_features=5000, 
#    vectorizer = TfidfVectorizer(lowercase=True, max_features=5000)     
    
    col_vector_train = vectorizer.fit_transform(X_train[col_to_vectorize])
    col_vector_test = vectorizer.transform(X_test[col_to_vectorize])

    print("Vectorizer Vocabulary contains : %d terms" %(len(vectorizer.vocabulary_)) )
    print("First Vocabulary terms :", dict(list(vectorizer.vocabulary_.items() )[:10]) )
    
    return col_vector_train, col_vector_test, vectorizer


def do_nothing(tokens):
    return tokens






#################################################################################################################


def initialize_text_model(model_type, Nb_features, Nb_classes):
    '''
    define which model to initialize for text data.
    Add other elif clausses to add other models initialization functions
    '''
    
    if model_type == 'NN':
        model = initialize_NN(Nb_features, Nb_classes)
    
    else:
        print("No model was initalized")
        model = None
        
    return model


def initialize_NN(Nb_features, Nb_classes):
    '''
    Initialize simple NN according to the data dimensions passed as arguments.
    '''
    
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    
    
    ## instantiate layers
    inputs = Input(shape = Nb_features, name = "input")
    
    dense1 = Dense(units = 256, activation = "relu",
                   kernel_initializer ='normal', name = "dense_1")
    
    dense2 = Dense(units = Nb_classes, activation = "softmax",      # for multiclass classification
                   kernel_initializer ='normal', name = "dense_2")
    
    
    ## link layers & model
    x = dense1(inputs)
    outputs = dense2(x)
    
    NN_clf = Model(inputs = inputs, outputs = outputs)
    
    
    ## define training process
    NN_clf.compile(loss = 'categorical_crossentropy',  
              optimizer = 'adam',                 
              metrics = ['accuracy'])  

    display(NN_clf.summary())
    
    return NN_clf
    

def save_model(model, name, path, doit = False):
    
#     from joblib import dump, load

    if doit:
        
        fitting_time = date_time()
        model_filename =  path + fitting_time + '_' + name + '.keras'

#         dump(model, model_filename)
        model.save(model_filename)
        print(f"Model saved as {model_filename}")

    else:
        print("model is not saved. Set doit = True to store the model locally. \n\n")
    
        

def reload_model(model_fullname, path, doit = False):
    
#     from joblib import dump, load
    import tensorflow as tf

    if doit:
        model_filename =  path + model_fullname 

#         reloaded_model = load(model, model_filename)
        reloaded_model = tf.keras.models.load_model(model_filename)
        print(f"Reloaded model from {model_filename}")
       
        return reloaded_model
        
    else:
        print("The model is NOT reloaded. Set doit = True to reload a model. \n")



#################################################################################################################


def preprocess_image_data(df, threshold, new_pixel_nb, path, output ='array', verbose = False):
    
    if ('productid' not in df.columns) or ('imageid' not in df.columns):
        print("Image data cannot be found from information on the dataframe. Try with another dataset.")
        return None
    
    import cv2
    
    t0 = time.time()
    
    img_array = np.empty((df.shape[0], new_pixel_nb * new_pixel_nb * 3), dtype = np.uint8)
    
    for i, idx in enumerate(df.index):
        
        # load image
        file = path + "image_" + str(df.loc[idx,'imageid'])+"_product_" \
                                               + str(df.loc[idx,'productid'])+".jpg"
        image = cv2.imread(file)
        
        # crop image 
        cropped_image = crop_image(image, threshold = threshold)
        
        # resize image (downscale)
        resized_image = cv2.resize(cropped_image, (new_pixel_nb, new_pixel_nb))
    
        # vectorize image (3D -> 1D) and append to general array
        img_array[i,...] = resized_image.reshape(new_pixel_nb*new_pixel_nb*3)
        
        if verbose:
            checkpoints = [1000,2000,3000,4000]
            if ((i in checkpoints) or i%5000 ==0):
                print("%d images at time %0.2f minutes" %(i, ((time.time()-t0)/60) ) )

                
    ## prepare dataframe with vector images
    df_vectors = pd.DataFrame(data = img_array)
    
    df_vectors.index = df.index
    
    column_names = []
    for j in range(new_pixel_nb*new_pixel_nb*3):
        column_names.append('px_'+str(j))
    df_vectors.columns = column_names

    
    t1 = time.time()
    if verbose:
        #print("Vectorization of %d images takes %0.2f seconds" %(df.shape[0],(t1-t0)) )
        print("Vectorization of %d images takes %0.2f minutes" %(df.shape[0],((t1-t0)/60)) )                

    if output == 'dataframe':
        return df_vectors
    elif output == 'array':
        return img_array
    

def crop_image(image, threshold):

    # Calculate the boundaries at which the RGB threshold is touched
    left_boundary = find_left_boundary(image, threshold)
    right_boundary = find_right_boundary(image, threshold)
    top_boundary = find_top_boundary(image, threshold)
    bottom_boundary = find_bottom_boundary(image, threshold)

    # crop image smallest square possible (including all boundaries inside)
    cropped_image = crop_square(image, left_boundary, right_boundary, top_boundary, bottom_boundary)

    return cropped_image



def find_left_boundary(image_array, threshold):
    height, width, _ = image_array.shape

    left_boundary = None
    for col in range(width):
        
        if np.any(image_array[:,col,:] < threshold):
            left_boundary = col
            break

    if left_boundary is None:
        left_boundary = 0

    return left_boundary

def find_right_boundary(image_array, threshold):
    height, width, _ = image_array.shape

    right_boundary = None
    for col in range(width - 1, -1, -1):
        
        if np.any(image_array[:,col,:] < threshold):
            right_boundary = col
            break

    if right_boundary is None:
        right_boundary = width - 1

    return right_boundary

def find_top_boundary(image_array, threshold):
    height, width, _ = image_array.shape

    top_boundary = None
    for row in range(height):
    
        if np.any(image_array[row,:,:] < threshold):
            top_boundary = row
            break

    if top_boundary is None:
        top_boundary = 0

    return top_boundary

def find_bottom_boundary(image_array, threshold):
    height, width, _ = image_array.shape

    bottom_boundary = None
    for row in range(height - 1, -1, -1):
        
        if np.any(image_array[row,:,:] < threshold):
            bottom_boundary = row
            break

    if bottom_boundary is None:
        bottom_boundary = height - 1

    return bottom_boundary



def crop_square(image_array, left, right, top, bottom):
    cropped_width = right - left + 1
    cropped_height = bottom - top + 1

    # Calculate the side length of the largest square that fits all boundaries
    side_length = max(cropped_width, cropped_height)

    horizontal_pad = (side_length - cropped_width) // 2
    vertical_pad = (side_length - cropped_height) // 2

    left_new = max(0, left - horizontal_pad)
    right_new = min(image_array.shape[1] - 1, right + horizontal_pad)
    top_new = max(0, top - vertical_pad)
    bottom_new = min(image_array.shape[0] - 1, bottom + vertical_pad)
    
    
    ## verify if vertical dimension iqueals horizontal dimension, and correct:
    if (right_new - left_new) > (bottom_new - top_new):
        if top_new > 0:
            top_new = top - vertical_pad - 1
        elif bottom_new < image_array.shape[0] - 1:
            bottom_new = bottom + vertical_pad + 1
    elif (right_new - left_new) < (bottom_new - top_new):
        if left_new > 0:
            left_new = left - horizontal_pad - 1
        elif right_new < image_array.shape[1] - 1:
            right_new = right + horizontal_pad + 1
    
 
    cropped_image = image_array[top_new : bottom_new+1, left_new : right_new+1, :]
    return cropped_image



def get_image_data(df_image_train, df_image_test, pixel_per_side, scale = None):
    '''
    df_image_train contains the pixel dataframe, only that. 
    One image per row (flattened) 1 feature = 1 pixel.
    This is apreprocessed dataframe.
    Same for the df_image_test.
    '''
    
    
    ## reshape to have 4D- matrices (nb_images, width, height, depth) and renormalize.

    N_img_train = df_image_train.shape[0]
    N_img_test = df_image_test.shape[0]
    N_px = pixel_per_side
    N_ch = 3

#     XX_train = df_image_train.to_numpy().reshape((N_img_train, N_px, N_px, N_ch))
#     XX_test = df_image_test.to_numpy().reshape((N_img_test, N_px, N_px, N_ch))
    XX_train = df_image_train.reshape((N_img_train, N_px, N_px, N_ch))
    XX_test = df_image_test.reshape((N_img_test, N_px, N_px, N_ch))

    

    ## Re normalize pixels intensity range to [0,1]
    if scale is not None:
        XX_train = XX_train / scale
        XX_test = XX_test / scale

    ## pack data
    image_data = {'train' : XX_train,
                  'test'  : XX_test }#,
#                   'train_index' : df_image_train.index,
#                   'test_index'  : df_image_test.index}
    
    return image_data


#################################################################################################################

    
def initialize_image_model(model_type, image_shape, Nb_classes):
    '''
    define which model to initialize for text data.
    Add other elif clausses to add other models initialization functions
    '''
    
    if model_type == 'CNN':
        model = initialize_CNN(image_shape, Nb_classes)
    
    else:
        print("No model was initalized")
        model = None
        
    return model



def initialize_CNN(image_shape, Nb_classes):
        
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv2D 
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Dropout 
    from tensorflow.keras.layers import Flatten

    ## instantiate layers
    inputs = Input(shape = image_shape, name = "input")

    
    ## first convolution layers
    C1_layer = Conv2D(filters = 8,
                         kernel_size = (3, 3),
                         padding = 'same',  # better than 'valid'
                         activation = 'relu')

    P1_layer = MaxPooling2D(pool_size = (2, 2))

    ## second convolution layer
    C2_layer = Conv2D(filters = 32,
                         kernel_size = (5, 5),
                         padding = 'valid',  # to shrink output size a bit
                         strides = (2,2),
                         activation = 'relu')

    P2_layer = MaxPooling2D(pool_size = (2, 2))

    ## drop out and flattening:
    Drp1_layer = Dropout(rate = 0.4)
    Flt_layer = Flatten()

    ## dense layers:
    D1_layer = Dense(units = 512,
                        activation = 'relu')

    Drp2_layer = Dropout(rate = 0.7)

    D2_layer = Dense(units = 128,
                        activation = 'relu')

    output_layer = Dense(units = Nb_classes,
                         activation='softmax')

    
    ## link layers & model

    x=C1_layer(inputs)
    x=P1_layer(x)

    x=C2_layer(x)
    x=P2_layer(x)

    x=Drp1_layer(x)
    x=Flt_layer(x)

    x=D1_layer(x)
    x=Drp2_layer(x)
    x=D2_layer(x)

    outputs=output_layer(x)

    CNN_clf = Model(inputs = inputs, outputs = outputs)
    
    
    ## define training process
    CNN_clf.compile(loss='categorical_crossentropy',
              optimizer='adam',                
              metrics=['accuracy'])

    
    ## display architecture
    CNN_clf.summary()
    
    
    return CNN_clf




###########################################################################################################
##### Fusion model ###########################

def remove_classification_head(parent_model):
    
    from tensorflow.keras.models import Model

    x = parent_model.layers[-2].output

    headless_model = Model(inputs = parent_model.input, outputs = x)

    display(headless_model.summary())
        
    return headless_model



def initialize_fusion_model(model_type, params):
    '''
    define which model to initialize for text data.
    Add other elif clausses to add other models initialization functions
    '''
    
    if model_type == 'NN':
        model = initialize_fusion_NN(params)
        
    else:
        print("No model was initalized")
        model = None
        
    return model


def initialize_fusion_NN(params): #(Nb_features, Nb_classes):
    '''
    Initialize simple NN according to the data dimensions passed as arguments.
    '''
    
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dropout 
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    
    Nb_features = params['Nb_features']
    Nb_classes = params['Nb_classes']
    
    ## instantiate layers
    inputs = Input(shape = Nb_features, name = "input")
    
    norm_0 = BatchNormalization() #epsilon=1e-06, mode=0, momentum=0.9, weights=None
    
#     dense1 = Dense(units = 256, activation = "relu",
#                    kernel_initializer ='normal', name = "dense_1")
    dense1 = Dense(units = 256,
               kernel_initializer ='normal', name = "dense_1")

    norm_1 = BatchNormalization() #epsilon=1e-06, mode=0, momentum=0.9, weights=None

    act_1 = Activation('relu')
    
    drop_12 = Dropout(rate = 0.7)

#     dense2 = Dense(units = 512, activation = "relu",
#                    kernel_initializer ='normal', name = "dense_2")
    dense2 = Dense(units = 256,
                   kernel_initializer ='normal', name = "dense_2")
        
    norm_2 = BatchNormalization() #epsilon=1e-06, mode=0, momentum=0.9, weights=None
  
    act_2 = Activation('relu')

    drop_23 = Dropout(rate = 0.5)
    
    dense3 = Dense(units = Nb_classes, activation = "softmax",      # for multiclass classification
                   kernel_initializer ='normal', name = "dense_3")

#     drop_3o = Dropout(rate = 0.05)
    
    ## link layers & model

#     x = dense1(inputs)
    
    x = norm_0(inputs)
    x = dense1(x)
    x = norm_1(x)
    x = act_1(x)

    x = drop_12(x)
    x = dense2(x)
    x = norm_2(x)
    X = act_2(x)

    x = drop_23(x)
#     x = dense3(x)
#     outputs = drop_3o(x)
    outputs = dense3(x)
    
    
    NN_clf = Model(inputs = inputs, outputs = outputs)
    
    
    ## define training process
#     NN_clf.compile(loss = 'categorical_crossentropy',  
#               optimizer = 'adam',                 
#               metrics = ['accuracy'])  

    display(NN_clf.summary())
    
    return NN_clf


# def get_schedule(epoch, lr, lr0, epochs, kind = 'exp'):
#     '''
#     define the type of scheduler function f(epoch, lr) to return
#     '''
    
#     if kind == 'rational':
#         decay_rate = lr0 / epochs  # good initial estimation. It can be set to another value.
        
#         def rational_decay(epoch, lr):
#             return lr/(1+decay_rate*epoch)

#         return rational_decay
    
#     if kind == 'step':
#         dropRate = 0.5  # float in (0,1], amount by which the learning rate is moified each time.
#         epochRate = 5   # int >= 1, how often to modify the learning rate.
        
#         def step_decay(epoch, lr):
#             return lr0 * math.pow(dropRate, math.floor( (1+epoch)/epochRate ) )
    
#         return step_decay
    
#     if kind == 'exp':
#         initial_wait = 10  # initial waitin Nb of epochs before starting to decrease lr.
#         decay_rate = -0.1  # float in [-inf,0] 
        
#         def exp_decay(epoch, lr):
#             if epoch < initial_wait:
#                 return lr
#             else:
#                 return lr * tf.math.exp(decay_rate)
        
#         return exp_decay
    

def get_rational_schedule(decay_rate):
    '''
    decay_rate = lr_initial / Nb_epochs is a good initial estimation. 
    But it can be adjusted to another value.
    '''
    ## define scheduler function
    def rational_decay(epoch, lr):
        return lr/(1+decay_rate*epoch)

    ## return function
    return rational_decay


def get_step_schedule(decay_drop, decay_freq):
    
    ## define scheduler function    
    def step_decay(epoch, lr):
        if (epoch % decay_freq) != 0 or epoch == 0:
            return lr
        else:
            return lr*decay_drop 

    ## return function
    return step_decay
        
        
def get_exp_schedule(decay_rate, initial_wait):
    
    ## define scheduler function
    def exp_decay(epoch, lr):
        if epoch < initial_wait:
            return lr
        else:
            return lr * math.exp(-decay_rate)

    ## return function
    return exp_decay
    

def plot_lr_schedule(schedules, names, lr0, epochs):
    
    art = sns.color_palette()
    labels = []

    plt.figure(figsize = (4.5,3.5))
    for i, schedule in enumerate(schedules):
        lr = lr0
        for epoch in range(epochs):
            lr = schedule(epoch, lr)
            plt.scatter(epoch, lr, color=art[i], s = 10)
        
            if epoch == 1:
                plt.scatter(epoch, lr, color=art[i], s = 10, label = names[i]) #, label='schedule_'+str(i+1) 
    
    plt.legend(loc = 'center right')
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
#     plt.ylim(0,0.2e-4)
    plt.show()
    
    return 

##########################################################################################################
## model evaluations

def plot_training_history(training_history, N_epochs, AccRange = None, LossRange = None):
    '''
    simple plot of the training and validation accuracy vs epochs
    '''
    
    x_epochs = np.arange(1,N_epochs + 1,1)

    train_acc = training_history.history['accuracy']
    val_acc = training_history.history['val_accuracy']

    train_loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']

    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    fig, axs = plt.subplots(1,2,figsize=(12,5))

    ax = axs[0]
    sns.lineplot(x = x_epochs, y = train_acc, marker = 'o', ax=ax, label = 'Training Accuracy')
    sns.lineplot(x = x_epochs, y = val_acc, marker = 'o', ax=ax, label = 'Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Evolution during training')
    ax.legend()#loc='right'
    ax.set_ylim(AccRange[0], AccRange[1]) if AccRange is not None else ax.set_ylim()
    
    ax = axs[1]
    sns.lineplot(x = x_epochs, y = train_loss, marker = 'o', ax=ax, label = 'Training loss')
    sns.lineplot(x = x_epochs, y = val_loss, marker = 'o', ax=ax, label = 'Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss function')
    ax.set_title('Loss Evolution during training')
    ax.legend()#loc='right'
    ax.set_ylim(LossRange[0], LossRange[1]) if LossRange is not None else ax.set_ylim()
    
    return


def get_train_test_accuracy(model, X_train, X_test, y_train, y_test):
    
    loss_train, accuracy_train = model.evaluate(X_train, y_train)
    loss_test, accuracy_test = model.evaluate(X_test, y_test)

    print("Train set accuracy = %0.3f and loss function = %0.2f" %(accuracy_train, loss_train) )
    print("Test  set accuracy = %0.3f and loss function = %0.2f" %(accuracy_test, loss_test) )

    return


    yy_pred_vectors = model.predict(X_test)

def get_confusionMatrix(y_test_vectors, y_pred_vectors, target_encoder, categories):

    ## reverse One-hot-encoding
    y_pred_class = y_pred_vectors.argmax(axis = 1)
    y_test_class = y_test_vectors.argmax(axis = 1)

    ## reverse label encoder
    y_pred = target_encoder.inverse_transform(y_pred_class)
    y_test_prdCode = target_encoder.inverse_transform(y_test_class)   # should be equal to y_test
    
    ## set categories
    categories = categories.tolist()

    y_pred_cat = pd.Categorical(y_pred, categories = categories)
    y_test_cat = pd.Categorical(y_test_prdCode, categories = categories)

    ## confusion matrix
    cm = pd.crosstab(y_test_cat, y_pred_cat, rownames=['Reality'], colnames=['Predictions'], dropna = False)

    return cm


def plot_confusionMatrix(cm):

    print(cm.shape)
    
    ## plot confusion matrix
    
    fig, ax = plt.subplots(figsize = (18,10))
    sns.heatmap(cm, annot = True, ax=ax, cmap='Greens', fmt ='d', vmin = 0, vmax = 100)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_tick_params(length = 0)
    ax.xaxis.set_label_position('top')

    return



def get_classificationReport(y_test_vectors, y_pred_vectors, target_encoder, categories):

    ## reverse One-hot-encoding
    y_pred_class = y_pred_vectors.argmax(axis = 1)
    y_test_class = y_test_vectors.argmax(axis = 1)

    ## reverse label encoder
    y_pred = target_encoder.inverse_transform(y_pred_class)
    y_test_prdCode = target_encoder.inverse_transform(y_test_class)   # should be equal to y_test
    
 
    ## classification report
    from sklearn.metrics import classification_report 

    cr_txt = classification_report(y_test_prdCode, y_pred)
    cr = classification_report(y_test_prdCode, y_pred, output_dict = True)

    cr.update({"accuracy": {"precision": None, 
                            "recall": None, 
                            "f1-score": cr["accuracy"], 
                            "support": cr['macro avg']['support']} })

    micro_cr = pd.DataFrame(cr).transpose().reset_index().rename(columns={'index': 'prdtypecode'}).iloc[:-3,:]
    macro_cr = pd.DataFrame(cr).transpose().reset_index().rename(columns={'index': 'metrics'}).iloc[-3:,:]

    return cr_txt, micro_cr, macro_cr


def plot_classificationReport(micro_cr):
    art = sns.color_palette()

    fig, axs = plt.subplots(1,2,figsize = (5.833,9),gridspec_kw={'width_ratios': [3.0, 2.0]})

    sns.heatmap(micro_cr.set_index('prdtypecode')[['precision', 'recall', 'f1-score']], annot = True, 
                cmap='viridis', vmin = 0, vmax = 1, ax = axs[0], cbar = False)
    
    sns.barplot(data = micro_cr, x = 'support', y='prdtypecode', color = 'grey', alpha = 0.75, ax = axs[1])


    axs[0].xaxis.set_ticks_position('top')
    axs[0].xaxis.set_tick_params(length = 0)
    axs[0].xaxis.set_label_position('top')
    axs[0].set_ylabel('Product Type Code')

    axs[1].set_xticks([0,500,1000,1500,2000])
    axs[1].set_xticklabels([0,'',1000,'',2000], fontsize=12)
    axs[1].yaxis.set_tick_params(labelleft=False)
    axs[1].set_ylabel('')
    axs[1].set_xlabel('Nb of observations')

    plt.subplots_adjust(wspace=0.02, hspace=0);
    
    return


def save_model_metrics(data, metric_type, model_name, path, timestamp):
    '''
    Save the data being iether the confusion matrix or the classififcation report.
    Chose 'metric_type' from the below list:
        'confusionMatrix'
        'classificationReport_txt'
        'classificationReport_df' (preferable option)
    '''
    
    filename = path + timestamp +'_' + model_name
    
    if metric_type == 'confusionMatrix':
        filename = filename + '_confusionMatrix.csv'  
        
        data.to_csv(filename, header = True, index = True)
        print(filename)    


    elif metric_type == 'classificationReport_df':
        filename = filename + '_classificationReport.csv'  
        
        data.to_csv(filename, header = True, index = True)
        print(filename)    

        
    elif metric_type == 'classificationReport_txt':
        filename = filename + '_classificationReport.txt'
        
        with open(filename, 'w') as file:
            file.write(data)
            print(filename)    

            



