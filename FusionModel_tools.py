## define date & time to print-out files

def date_time():
    '''
    get date and time in string format '_yymmdd_hhmm'
    at the moment the function is called.
    '''
    from datetime import date, datetime
    
    today = date.today()
    now = datetime.now() 

    return today.strftime("_%Y%m%d") +'_'+ now.strftime("%H%M")



def save(df_train, df_test, doit):
    
    if doit == True:
    
        splitting_time = date_time()

        train_filename = 'df_train' + splitting_time +'.csv'
        df_train.to_csv(train_filename, header = True, index = False)
        print("Train dataset exported as %s" %train_filename)

        test_filename = 'df_test' + splitting_time +'.csv'
        df_train.to_csv(test_filename, header = True, index = False)
        print("Test dataset exported as %s" %test_filename)
        
    else:
        print("Datasets were not saved locally. Set doit = True to save datasets")
        
        
def initialize_tesxt_model(model_type):
    '''
    define which model to initialize for text data.
    Add other elif clausses to add other models initialization functions
    '''
    
    if model_type = 'NN':
        model = initialize_NN()
    
    else:
        print("No model was initalized")
        model = None
        
    return model


def initialize_NN():
    return None







































