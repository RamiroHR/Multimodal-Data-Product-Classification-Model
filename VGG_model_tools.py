import numpy as np
import pandas as pd
import os 
import cv2
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


########################################################################""

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

        saving_time = date_time()

        for data, type_, name in zip(datasets, types, names):
            filename = saving_time + '_' + name 
            
            if type_ == 'dataframe':     
                filename = filename + '.csv'
                data.to_csv(path + filename, header = True, index = True)  # need index after train_test_split
                print("Saved dataset: %s" % (path+filename)) if verbose else None

            elif type_ == 'array':
                filename = filename + '.npy'
                np.save(os.path.join(path, filename), data)
                print("Saved dataset: %s" % (path+filename) ) if verbose else None
#                 loaded_data = np.load(save_path)
                
            elif type_ == 'sparseMatrix':
                filename = filename + '.npz'
                from scipy import sparse
                sparse.save_npz(os.path.join(path, filename), data)
                print("Saved sparseMatrix : %s" % (path+filename) ) if verbose else None
#                 your_matrix_back = sparse.load_npz("yourmatrix.npz")
                
#             elif type_ == 'transformer':
#                 filename = filename
#                 import joblib
#                 joblib.dump(data, os.path.join(path, filename))
#                 print("Saved transformer: %s" % (path+filename) ) if verbose else None
# #                 my_scaler = joblib.load('scaler.gz')

#             elif type_ == 'XLMatrix':
#                 filename = filename + '.npy'
#                 import joblib
#                 joblib.dump(data, os.path.join(path, filename))
#                 print("Saved large matrix: %s" % (path+filename) ) if verbose else None
# #                 my_matrix = joblib.load('matrix')

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

    
    
    
def get_items_by_processing_order(gen_dataset, X_subset, y_subset):

    new_index = []

    for i, file in enumerate(gen_dataset.filenames):
        img_id = int(file.split("_")[1])
        prd_id = int(file.split("_")[3][:-4])

        ## find item index in the dataframe
        idx = X_subset[ (X_subset['productid']==prd_id) & 
                        (X_subset['imageid']==img_id) ].index[0]
        
        new_index.append(idx)
        
    ## find the correct class of the product
    y_subset_sorted = y_subset.reindex(new_index, copy = True)
    X_subset_sorted_ids = X_subset.reindex(new_index, copy = True).loc[:,['imageid','productid']]
    
    processed_items = pd.concat([y_subset_sorted, X_subset_sorted_ids], axis = 1)
    
    return processed_items
    

def Reorder_subsets_by_VGG_processing_order(gen_dataset, X_subset, y_subset):

    new_index = []

    for i, file in enumerate(gen_dataset.filenames):
        img_id = int(file.split("_")[1])
        prd_id = int(file.split("_")[3][:-4])

        ## find item index in the dataframe
        idx = X_subset[ (X_subset['productid']==prd_id) & 
                        (X_subset['imageid']==img_id) ].index[0]
        
        new_index.append(idx)
        
    ## find the correct class of the product
    y_subset_sorted = y_subset.reindex(new_index, copy = True)
    X_subset_sorted = X_subset.reindex(new_index, copy = True)
        
    return X_subset_sorted, y_subset_sorted
    
    
    
def transform_target(y_train, y_val, y_test):
    '''
    transform target varibale as needed for the choosen model.
    '''

    ## Label encoder
    from sklearn.preprocessing import LabelEncoder
    target_encoder = LabelEncoder()
    
    y_train_encoded = target_encoder.fit_transform(y_train.squeeze())
    y_val_encoded = target_encoder.transform(y_val.squeeze())
    y_test_encoded = target_encoder.transform(y_test.squeeze())
    

    ## One Hot encoder
    from tensorflow.keras.utils import to_categorical

    yy_train = to_categorical(y_train_encoded, dtype = 'int') 
    yy_val = to_categorical(y_val_encoded, dtype = 'int')       
    yy_test = to_categorical(y_test_encoded, dtype = 'int')   
    
    
    return yy_train, yy_val, yy_test, target_encoder





def get_callbacks(check_path):
    
    callbacks = []
    
    callbacks.append( get_model_checkpoint(check_path) )
#     callbacks.append( get_early_stopping() )
    callbacks.append( get_reduceLRonPlateau() )
#     callbacks.append( LR_scheduler()[0] )  # returns a tuple, select only the scheduler
    
    return callbacks


def get_model_checkpoint(check_path):    
    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint_filepath = check_path #'./tmp_checkpoint/img_baseModel_vgg_refined/'
    checkpoint = ModelCheckpoint(
                                filepath=checkpoint_filepath,
                                verbose = 1,
                                save_weights_only=True,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True)
    return checkpoint
    
    
def get_early_stopping():
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor = 'val_loss',
                                   patience = 5,
                                   mode = 'min',
                                   restore_best_weights = True)
    return early_stopping
   
    
def get_reduceLRonPlateau():
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    lr_plateau = ReduceLROnPlateau(monitor = 'val_loss',
                                   patience=5,
                                   factor=0.7,
                                   verbose=1,
                                   mode='min',
                                   min_lr = 1e-10)
    return lr_plateau


def LR_scheduler():
    from tensorflow.keras.callbacks import LearningRateScheduler
    # schedule1 = get_rational_schedule(decay_rate = lr_0 / Nb_epochs)  # 
    schedule = get_step_schedule(decay_drop = 0.55, decay_freq = 2)  # 
    # schedule3 = get_exp_schedule(decay_rate = 0.4, initial_wait = 1)  #
    # sch_names = ['rat', 'step', 'exp']

    lr_scheduler.append(LearningRateScheduler(schedule, verbose = 1) )

    return lr_scheduler, schedule
    
    
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


def initialize_head_model(u1,u2,drop1,drop2, Nb_classes, input_shape):
    from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
    from tensorflow.keras.models import Model, Sequential

    head_model = Sequential()
    head_model.add( Flatten(input_shape = input_shape) )
    head_model.add( Dense(units = u1, activation = 'relu') )
    head_model.add( Dropout(rate = drop1) )
    head_model.add( Dense(units = u2, activation = 'relu') )
    head_model.add( Dropout(rate = drop2) )
    head_model.add( Dense(units = Nb_classes, activation='softmax') )

    return head_model()


def compile_model(head_model, lr_0):
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate = lr_0)

    head_model.compile(loss = 'categorical_crossentropy',  # because targets are one hot encoded
                      optimizer = optimizer,
                      metrics = ['accuracy'])

    return head_model


def create_checkpoint_folders(foldername, checkpoints_path = './subset/'):
    
    folder = checkpoints_path + foldername

    if not os.path.exists(folder):
        print('created folder')
        os.mkdir(folder)
    else:
        print(' exists!')

    return folder+'/'


def save_model(model, name, path, doit = False):
    
    if doit:
        
        fitting_time = date_time()
        model_filename =  path + fitting_time + '_' + name + '.keras'

        model.save(model_filename)
        print(f"Model saved as {model_filename}")

    else:
        print("model is not saved. Set doit = True to store the model locally. \n\n")
    

    
def reload_model(model_fullname, path, doit = False):
    
    import tensorflow as tf

    if doit:
        model_filename =  path + model_fullname 

        reloaded_model = tf.keras.models.load_model(model_filename)
        print(f"Reloaded model from {model_filename}")
       
        return reloaded_model
        
    else:
        print("The model is NOT reloaded. Set doit = True to reload a model. \n")

    
    
###########################################################""
##---------Evaluate method---------------##
    
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
    
    fig, ax = plt.subplots(figsize = (18,10))
    sns.heatmap(cm, annot = True, ax=ax, cmap='Greens', fmt ='d', vmin = 0, vmax = 100, 
                cbar_kws={"pad": 0.02, "aspect":30})
    
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_tick_params(length = 0)
    ax.xaxis.set_label_position('top')
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.xlabel('Predictions',fontsize=18)
    plt.ylabel('Reality', fontsize=16)
    
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


def plot_classificationReport_extract(micro_cr, nb_classes, nb_2show):
    art = sns.color_palette()

    fig, axs = plt.subplots(1,2,figsize = (5.833,9/nb_classes*nb_2show),gridspec_kw={'width_ratios': [3.0, 2.0]})

    sns.heatmap(micro_cr.set_index('prdtypecode')[['precision', 'recall', 'f1-score']], annot = True, 
                cmap='viridis', vmin = 0, vmax = 1, ax = axs[0], cbar = False)
    
    sns.barplot(data = micro_cr, x = 'support', y='prdtypecode', color = 'grey', alpha = 0.75, ax = axs[1])


    axs[0].xaxis.set_ticks_position('top')
    axs[0].xaxis.set_tick_params(length = 0)
    axs[0].xaxis.set_label_position('top')
    axs[0].set_ylabel('Category')

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

    
    
    
    
    
    
######################  Flow_from_directory  ###################################
# '''
# There asre some functions used in the context of implementing flow_from_directory on ImageDataGenerators
# '''


# def verify_subset_accuracy(VGG_model, val_dataset, X_val_subset, y_val_subset):
#     '''
#     val_dataset is an ImageDataGenerator().flow_from_directory() type of object.
#     X_val_subset and y_val_subset are the dataframes associated to the subset.
#     '''
    
# #     print("Evaluate method: Evaluate the accuracy from using the evaluate() method")    
# #     print("-----------------------------------------------------------------------")
# #     val_dataset.reset()
# #     loss_val, accuracy_val = VGG_model.evaluate(val_dataset)
# #     print("\t accuracy_val =", accuracy_val,'\n')

    
#     print("Predictions method: Evaluate the accuracy from the predictions")    
#     print("--------------------------------------------------------------")
#     t0 = time.time()
#     val_dataset.reset()  
#     yy_pred_vectors = VGG_model.predict(val_dataset)
#     t1 = time.time()
#     print("Predictions done in %0.2f seconds" %(t1-t0))
#     print("Output vector has shape:", yy_pred_vectors.shape, '\n')

#     ## transform to class index
#     y_pred_class = yy_pred_vectors.argmax(axis = 1)
    
#     ## map the predicted labels to their unique ids (folder names)
#     labels = (val_dataset.class_indices)
#     labels = dict((v,k) for k,v in labels.items())
#     y_pred = np.array([labels[k] for k in y_pred_class], dtype = int)


#     ## Check in which order files where evaluated 
#     print("First 10 files evaluated:")
#     display(val_dataset.filenames[:10])
    
#     print("Identify the true_class for each evaluated file, in the order of evaluation")
#     true_classes = np.empty(len(val_dataset.filenames), dtype = 'int')

#     # test_dataset.class_indices  # no classes inferred
#     # train_dataset.class_indices  # classes inferred from folder names
#     for i, file in enumerate(val_dataset.filenames):
#         img_id = int(file.split("_")[1])
#         prd_id = int(file.split("_")[3][:-4])

#         ## find item index in the dataframe
#         idx = X_val_subset[ (X_val_subset['productid']==prd_id) & 
#                            (X_val_subset['imageid']==img_id) ].index[0]

#         ## find the correct class of the product
#         true_classes[i] = y_val_subset.loc[idx,'prdtypecode']
# #         display(len(true_classes))
# #         true_classes[:930]


#     ## Build classification report
#     print("Classification report")
#     from sklearn.metrics import classification_report 

#     cr = classification_report(true_classes, y_pred, output_dict = True)
#     # cr = classification_report(y_test_prdCode, y_pred, output_dict = True)
#     cr.update({"accuracy": {"precision": None, 
#                                "recall": None, 
#                              "f1-score": cr["accuracy"], 
#                               "support": cr['macro avg']['support']}})

#     micro_CNN = pd.DataFrame(cr).transpose().reset_index().rename(columns={'index': 'prdtypecode'}).iloc[:-3,:]
#     macro_CNN = pd.DataFrame(cr).transpose().reset_index().rename(columns={'index': 'metrics'}).iloc[-3:,:]

#     display(macro_CNN)
#     accuracy_val_2 = macro_CNN[macro_CNN['metrics']=='accuracy']['f1-score'].values[0]

#     print("Verify that both measures of accuracies are compatible:")
#     print("\t evaluate_method_accuracy = ", accuracy_val)
#     print("\t predict_method_accuracy  = ", accuracy_val_2)

