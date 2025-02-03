import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def date_time():
    from datetime import date, datetime
    
    today = date.today()
    now = datetime.now() 

    return today.strftime("%Y%m%d") +'_'+ now.strftime("%H%M")



def concat_class_code(product_class):
    '''
    Concatenate the product class name with the code of the class as className_classCode
    Pad with extra '_' for codes with less than 4 digits
    '''
    
    product_class['class_code'] = [class_ +'_'+ str(code) if code not in [10,40,50,60] 
                                   else class_ +'___'+ str(code)
                                   for class_, code in zip(product_class['prodtype'],product_class['prdtypecode']) ]    



def plot_counts_per_category(df, product_class):
    ## Counts per cathegory in the whole train dataset.
    
    fsizeL = 15

    class_codes = df['prdtypecode'].value_counts().index
    class_labels = [ product_class.loc[product_class['prdtypecode']==code,'class_code'].iloc[0] for code in class_codes]


    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))
    sns.countplot(y = 'prdtypecode', data = df , order = class_codes) # or order = class_codes

    ax.set_yticklabels(class_labels)    # set product labels

    plt.ylabel('Product Class', fontsize = fsizeL)
    plt.xlabel('Counts', fontsize = fsizeL)
    plt.title('Number of products per category', fontsize = fsizeL )
    plt.show();


def plot_missing_description_distribution(df, product_class):
    
    ## subset with missing description
    df_missing = df[df['description'].isna()]
    class_codes = df_missing['prdtypecode'].value_counts().index
#     class_labels = [ product_class.loc[product_class['prdtypecode']==code,'prodtype'].iloc[0] for code in class_codes]
    class_labels = [ product_class.loc[product_class['prdtypecode']==code,'class_code'].iloc[0] for code in class_codes]


    ## plot
    fsizeL = 15
    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))

    sns.countplot(y = 'prdtypecode', data = df_missing , order = class_codes)#df['prdtypecode'].value_counts().index

    # Set product labels
    ax.set_yticklabels(class_labels)

    plt.ylabel('Product Class', fontsize = fsizeL)
    plt.xlabel('Counts', fontsize = fsizeL)
    plt.title("Number of products with missing 'description'", fontsize = fsizeL )
    plt.show();
    
        
            
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


def get_html_tags():
    
    #define common tags to search for:
    html_top_tags = ['&nbsp','ol','i','u','s','sub','sup','em','br', 'b', 'div', 'p', 'a', 'img', 'ul', 'li',\
                         'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'table', 'tr', 'td', 'th']
    
    html_fr_special_letters = ['&eacute;','&egrave;','&ecirc;','&agrave;','&ccedil;','&euml;','&icirc;','&iuml;','&ocirc;',\
                               '&ucirc;','&oelig;']

    html_tags = html_top_tags + html_fr_special_letters
    
    return html_tags
    

def has_html_tag(text):
    import re
    
    html_tags = get_html_tags()
    
    cnt = 0
    for tag in html_tags:
        pattern = '<' + tag
        result  = re.search(pattern, text)
        if result:
            cnt += 1
      
    return True if (cnt >= 1) else False
    

def get_html_encoding_proportion(df, drop = False):
    
    df['has_html'] = [ has_html_tag(text) for text in df['title_descr'] ]
       
    bar_labels = [ str(round(val*100, 2))+' %' for val in df['has_html'].value_counts(normalize=True).values]
    bar_labels
    
    ## plot
    f, ax = plt.subplots(1,1, figsize = (6,5) )
    sns.countplot(x = 'has_html', data = df) 

    ## anotations
    fsize = 14
    ax.bar_label(ax.containers[0], labels = bar_labels, label_type='center', fontsize = fsize)
    plt.ylabel('Counts', fontsize = fsize)
    plt.xlabel("Does 'title_descr' feature have html encoding?", fontsize = fsize)
    plt.xticks(fontsize = fsize-1.5)
    plt.yticks(fontsize = fsize-1.5);
    plt.show()
    
    # drop variable (optional)
    if drop:
        df.drop('has_html', axis = 1, inplace = True)

        

def html_parsing(df, col_to_parse, verbose = False):
    '''
    HTML parse and lower case text content in col_to_parse
    '''
    from bs4 import BeautifulSoup

    import warnings
    from bs4 import MarkupResemblesLocatorWarning #GuessedAtParserWarning
    warnings.filterwarnings('ignore', category = MarkupResemblesLocatorWarning)
    
    
    t0 = time.time()
        
    df[col_to_parse] = [BeautifulSoup(text, "lxml").get_text().lower() for text in df.loc[:,col_to_parse]]  #lxml, html.parser

    t1 = time.time()
    
    if verbose: 
        print(f"Column '{col_to_parse}' has been successfully HTML parsed and decapitalized.")
        print("\t HTML parsing takes %0.2f seconds \n" %(t1-t0))
        

        
        
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

        cnt_fr, cnt_en = 0, 0
        
        for i, idx in zip( range(len(languages_probas)), df.index):    
            if languages_probas[i][1] < 0.9999:
                languages_probas[i] = identifier_2.classify(df.loc[idx,text_col]) ##### error
                
                if languages_probas[i][0] == 'fr':
                    cnt_fr = cnt_fr + 1
                elif languages_probas[i][0] == 'en':
                    cnt_en = cnt_en + 1        
                
                
    time_2 = time.time()

    
    ## save detection in dataframe
    df['language'] = list(np.array(languages_probas)[:,0])
    
    if get_probs:
        df['lang_prob'] = [float(p) for p in np.array(languages_probas)[:,1]]
    
    
    if verbose:
        print("Main language detection takes %0.2f minutes." %((time_1 - time_0)/60) )
        if correct:
            print("\t Correction takes %0.2f seconds to crrect %d low confidence detections" %((time_2 - time_1), cnt_fr+cnt_en) )
            print(f"\t A total of {cnt_fr} items have been reclassified as FR and {cnt_en} as EN for simplicity.\n")
    return


def plot_language_distribution(df, Nb_2show):
    
#     display(df['language'].value_counts(normalize = False).head(7))

    lg_order = df['language'].value_counts(normalize = False).head(Nb_2show).index

    lg_counts = df['language'].value_counts(normalize = True).head(Nb_2show).values*100
    lg_pct = [str(round(count,2))+' %' for count in lg_counts]
    
    f, ax = plt.subplots(1,1, figsize = (6,5) )
    sns.countplot(x = 'language', data = df, order = lg_order) 

    ## annotations
    fsize = 14
    ax.bar_label(ax.containers[0], labels = lg_pct, label_type='edge', fontsize = fsize-4)
    plt.ylabel('Counts', fontsize = fsize) 
    plt.xlabel("Language", fontsize = fsize) 
    plt.xticks(fontsize = fsize-1.5) 
    plt.yticks(fontsize = fsize-1.5) 
    plt.yscale("log")
    plt.show();
        


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
    


def get_category_top_words(df, product_class, Nb_top_words):
    
    category_tokens = create_category_pool(df, product_class)
    
    product_class = get_top_words(category_tokens, product_class, Nb_top_words)
    
    
    
def create_category_pool(df, product_class):
    cat_tokens = {}
    for categ in product_class['prdtypecode']:
        token_pool = []

        for token_list in df[df['prdtypecode'] == categ]['lemma_tokens']:
            token_pool += token_list  

        cat_tokens[str(categ)] = token_pool

    return cat_tokens


def get_top_words(cat_tokens, product_class, Nb_top_words):
    import collections

    N_most_common = Nb_top_words  ## around 5-15 enough
    common_word_dict = {}

#     for categ, pool in zip(product_class['prdtypecode'],product_class['token_pool']):
    for categ, pool in cat_tokens.items():
        
        ## get top words in increasing order of counts
        token_counter = collections.Counter(pool).most_common()

        ## remove words of 1 or 2 letters that may have been left if we didn't remove the stop words from 
        ## the appropiate language (this is the case for the less represented languages).
        common_word_dict[categ] = [item[0] for item in token_counter if len(item[0])>2]  

        ## Store the N first words
        common_word_dict[categ] = common_word_dict[categ][:N_most_common]

    ## Save into dataframe
    product_class['common_words'] = product_class['prdtypecode'].apply(lambda x: common_word_dict.get(str(x)))
        
    return product_class


    
def display_category_top_words(product_class, to_show = 'all'):
    from pandas import option_context  ## to set the length of the dataframe column to display
    
    if to_show == 'all':
        df2 = product_class
    else:
        df2 = product_class[ product_class['prdtypecode'].isin(to_show) ]
    
    ## selec some categories (image for report)
    with option_context('display.max_colwidth', 1000):
        display(df2.loc[:,['class_code','common_words']]) 


        
        
def get_token_length(df, col_with_tokens, col_with_length, verbose = False):
    '''
    Creates a new variable measuring the number of tokens in column col_with_tokens
    '''
    t0 = time.time()
    
    df[col_with_length] = [len(token_list) for token_list in df[col_with_tokens] ]
    
    t1 = time.time()
    
    if verbose:
        print("Token counting takes %0.2f seconds. \n" %(t1-t0))




def plot_token_len_distribution(df, xRange = None, qqplot = True, quantiles = False):
    '''
    Plot  
    '''
    sns.set(font_scale = 1.25)
    
    # Figure composed of two main matplotlib.Axes objects (ax_box and ax_hist)
    fsize = 14
    fig, (ax_box, ax_hist) = plt.subplots(2, figsize = (8,7), sharex=True, 
                                          gridspec_kw={"height_ratios": (.1, .9)})

    ## Box plot
    sns.boxplot(x = df.text_token_len, ax = ax_box)
    ax_box.set(xlabel='boxplot representation')
    
    
    ## hist plot
    sns.histplot(x = df.text_token_len, ax = ax_hist, binwidth = 2)#, kde = True
    ax_hist.set_xlabel('Number of words')
    ax_hist.set_ylabel('Number of products')
    if xRange is not None:
        ax_hist.set_xlim([xRange[0] , xRange[1]])

        
    ## Add quantiles and extreme values markers on main plot
    if quantiles == True:
        # quantile values
        q1, q2, q3 = df['text_token_len'].quantile(q=[0.25,0.5,0.75])
        iqr = q3-q1
        
        ax_hist.axvline(x=q1, ymin= 0, ymax = 0.1, c='k', ls = '--', label='Q1, Q2, Q3')
        ax_hist.axvline(x=q2, ymin= 0, ymax = 0.1, c='k', ls = '--')
        ax_hist.axvline(x=q3, ymin= 0, ymax = 0.1, c='k', ls = '--')
        ax_hist.axvline(x=q3+1.5*iqr, ymin= 0, ymax = 0.1, c='r', ls = ':', label='Q3 + 1.5 * IQR')
        ax_hist.legend(bbox_to_anchor = (0.35,1), fontsize = 10)
   

    ## Define the position and size of the inset axes with left and bottom shifts
    if qqplot == True:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax_hist,
                              width="45%",
                              height="45%",
                              loc='upper right',
                              bbox_to_anchor=(-0.033,-0.003,1,1),
                              bbox_transform=ax_hist.transAxes)  
                              # Adjust these values for left and bottom shifts 
                              # Otherwise use borderpad = 1.1 instead of the bbox arguments.

        import statsmodels.api as sm
        sm.qqplot(df['text_token_len'], fit=True, line='45', label ='text_token_len', ax = inset_ax, ms = 5)
        #inset_ax.title('Variable normamility test', fontsize=11) #, fontweight='bold'
        #inset_ax.legend(loc ='lower right')
        inset_ax.set_xlabel('Theoretical Quantiles', fontsize = 13)
        inset_ax.set_ylabel('Sample Quantiles', fontsize = 13)
        inset_ax.grid(True, linestyle='--')
    
    
    plt.show()
    sns.set()  # back to normal



def plot_token_len_distribution_per_category(df, product_class):
    
    fsizeL = 15

    order = df[['prdtypecode','text_token_len']].groupby('prdtypecode')['text_token_len'].mean().sort_values(ascending=False)
    str_order_index = [str(idx) for idx in order.index]

    df_plot = df.copy()
    df_plot['prdtypecode'] = df_plot['prdtypecode'].astype(str)
    df_plot['text_token_len'] = df_plot['text_token_len'].astype(int)

    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,10))
    sns.boxplot(x ='text_token_len' , y ='prdtypecode', data = df_plot , order = str_order_index)

    # set product labels
    class_codes = order.index
    class_labels = [ product_class.loc[product_class['prdtypecode']==code,'class_code'].iloc[0] for code in class_codes]
    ax.set_yticklabels(class_labels);

    plt.title("Token-length Distribution per Category", fontsize = 13)
    plt.ylabel("Product category", fontsize = 13)
    plt.xlabel("Number tokens in text data", fontsize = 13);



def run_anova_test(df, num_col, cat_col):

    # Import the library
    import statsmodels.api

    ## Realize the ANOVA test and display the results
    ## Try 'title_char_len', 'descr_char_len' or 'text_token_len'
    to_test_correlation = num_col + ' ~ ' + cat_col
    result = statsmodels.formula.api.ols( to_test_correlation , data = df).fit()
    table = statsmodels.api.stats.anova_lm(result)

    # get p-val an evaluate criteria
    display(table)
    p_val_anova = table.loc['prdtypecode','PR(>F)']

    # evaluation
    alpha = 0.05
    def p_val_evaluation(p_val, alpha):
        if p_val >= alpha:
            print("H0 is not rejected")
        else:
            print("H0 is rejected, H1 is accepted")

    p_val_evaluation(p_val_anova,alpha)
    print("ANOVA p-value = ", p_val_anova)



























