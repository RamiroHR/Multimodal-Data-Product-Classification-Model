import time
import numpy as np
import pandas as pd
import os 

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


def classification_reports_difference(cr1, cr2):
    
    cr_diff = cr1.set_index('prdtypecode').subtract(cr2.set_index('prdtypecode')).reset_index()
    
    ## plot cr_diff
    art = sns.color_palette()

    fig, axs = plt.subplots(1,2,figsize = (5.833,9),gridspec_kw={'width_ratios': [3.0, 2.0]})

    sns.heatmap(cr_diff.set_index('prdtypecode')[['precision', 'recall', 'f1-score']], annot = True, 
                cmap='RdBu', vmin = -0.1, vmax = 0.1, ax = axs[0], cbar = False)

    sns.barplot(data = cr1, x = 'support', y='prdtypecode', color = 'grey', alpha = 0.75, ax = axs[1])

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
    
    return None



################# Base model: NN  --------------------------

def compile_fusion_model(model, lr_0):
    
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate = lr_0)

    model.compile(loss = 'categorical_crossentropy',  # because targets are one hot encoded
                optimizer = optimizer,
                    metrics = ['accuracy'])
    
    return model


def get_callbacks(checkpoint_path):
    
    callbacks = []
    
    callbacks.append( get_model_checkpoint(checkpoint_path) )
#     callbacks.append( get_early_stopping() )
    callbacks.append( get_reduceLRonPlateau() )
#     callbacks.append( LR_scheduler()[0][0] )  # returns a tuple, the 0th element can be a list, select only the schedule to use
    
    return callbacks


def get_model_checkpoint(checkpoint_path):    
    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(
                                filepath=checkpoint_path,
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
                                   cooldown=1,
                                   min_lr = 1e-10)
    return lr_plateau


def LR_scheduler():
    from tensorflow.keras.callbacks import LearningRateScheduler
        
    ## Try different functions for the learning rate decay:
    
    schedules = []
    # schedules.append(get_rational_schedule(decay_rate = lr_0 / Nb_epochs))  # 
    schedules.append(get_step_schedule(decay_drop = 0.55, decay_freq = 10))  # (0.55,3) previous values
    # schedules.append(get_exp_schedule(decay_rate = 0.4, initial_wait = 1))  #
    # sch_names = ['rat', 'step', 'exp']

    lr_scheduler = []
    for schedule in schedules:
        lr_scheduler.append(LearningRateScheduler(schedule, verbose = 1) )

    return lr_scheduler, schedules
    
    
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

    fig, axs = plt.subplots(1,2,figsize=(12,4.5))

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



#######  model evaluation   #############


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

            
def create_checkpoint_folders(foldername, checkpoints_path = './subset/'):
    
    folder = checkpoints_path + foldername

    if not os.path.exists(folder):
        print('created folder')
        os.mkdir(folder)
    else:
        print(' exists!')

    return folder+'/'