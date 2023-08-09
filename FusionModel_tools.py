import time
import numpy as np


def date_time():
    '''
    get date and time in string format '_yymmdd_hhmm'
    at the moment the function is called.
    '''
    from datetime import date, datetime
    
    today = date.today()
    now = datetime.now() 

#     return today.strftime("%Y%m%d")[2:] +'_'+ now.strftime("%H%M")
    return today.strftime("%Y%m%d")[2:] + now.strftime("%H%M")



def save(dataframes, names,  path, doit = False, verbose = True):
    '''
    Save each dataframe in dataframes with the respective name in names.
    Save at the specified path.
    '''
          
    if doit == True:

        splitting_time = date_time()

        for df, name in zip(dataframes, names):
            
            filename = splitting_time + '_' + name + '.csv'
            df.to_csv(path + filename, header = True, index = True)  # need index after train_test_split
            print("Saved dataset: %s" %filename) if verbose else None
            
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
    
    
#     ## save preprocessed dataframe ?
#     save(dataframes = [df], names = [df_name], path = './Preprocessed_data/',\
#          doit = save_preprocessed, verbose = True)
    
    
    ##### data transformation #####
    
    ## Scale text_token_len
    
    ## One hot encode language
    
    ## title_descr TFIDF vectorizer

    
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
    
    t0 = time.time()
    
    with warnings.catch_warnings():     ## disable warnings when BeautifulSoup encounters just plain text.
        warnings.simplefilter("ignore")
        
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

    ## transform feature vairables
    X_transformed_train, X_transformed_test, X_transformer = transform_features(X_train, X_test)
    
    ## transform target variable
    y_transformed_train, y_transformed_test, y_transformer = transform_target(y_train, y_test)
    
    ## pack datasets
    text_data = {'X_train' : X_transformed_train,
                 'X_test'  : X_transformed_test,
                 'y_train' : y_transformed_train,
                 'y_test' : y_transformed_test}
    
    ## pack tranformers trained
    text_transformers = {'X_transformer' : X_transformer,
                         'y_transfomer'  : y_transformer}
    
    return text_data, text_transformers
    
    
        
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
    transformers = {'scaler' : scaler,
                   'encoded': encoder,
                   'vectorizer': vectorizer}
    
    
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

    vectorizer = TfidfVectorizer(tokenizer = do_nothing, lowercase=False) #max_features=5000, 

    col_vector_train = vectorizer.fit_transform(X_train[col_to_vectorize])
    col_vector_test = vectorizer.transform(X_test[col_to_vectorize])

    return col_vector_train, col_vector_test, vectorizer


def do_nothing(tokens):
    return tokens


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

    yy_train = y_train_encoded
    yy_test = y_test_encoded
    
    return yy_train, yy_test, target_encoder




####################################################################################################################


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
    inputs = Input(shape = Nb_features, name = "Input")
    
    dense1 = Dense(units = 512, activation = "ReLu", \
                   kernel_initializer ='normal', name = "Dense_1")
    
    dense2 = Dense(units = Nb_classes, activation = "softmax", \
                   kernel_initializer ='normal', name = "Dense_2")
    
    
    ## link layers & model
    
    x = dense1(inputs)
    outputs = dense2(x)
    
    NN_clf = Model(inputs = inputs, outputs = outputs)
    
    return NN_clf
    





































