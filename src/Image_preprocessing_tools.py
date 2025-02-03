import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import cv2


def date_time():
    from datetime import date, datetime
    
    today = date.today()
    now = datetime.now() 

    return today.strftime("%Y%m%d") +'_'+ now.strftime("%H%M")


def show_images(df, index = [], verbose = True):

    #sns.set_style("darkgrid")
    sns.set_style("white")

    fig, ax = plt.subplots(1, 3 , figsize=(4*len(index), 3) ) 

    for i, idx in enumerate(index):
        file1 = "../datasets/image_train/image_"+str(df.loc[idx,'imageid'])+"_product_"+str(df.loc[idx,'productid'])+".jpg"

        if verbose:
            print(file1)
    
        image1 = np.int64(cv2.imread(file1))
        
        #fig = plt.imshow(image[:,:,::-1])
        #fig = plt.axis("off")
        ax[i].imshow(image1[:,:,::-1])
        ax[i].grid(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        #ax.grid(color='w', linewidth=2)
        #ax[0].set_frame_on(False)0
        #ax[1].set_frame_on(True)
        #ax[2].set_frame_on(True)
    
    plt.show()    
    sns.set() #back to normal

    
def show_image_from_category(df, product_class, category, verbose = False):
    
    ## if the ctagory name is passed
    if category not in product_class['prdtypecode'].to_list():
        category = product_class[product_class['prodtype'] == category]['prdtypecode'].values[0]

    ## directly here when the category code is passed
    indexes = df[ df['prdtypecode'] == category].index
    
    display(product_class[ product_class['prdtypecode'] == category].loc[:,['prdtypecode', 'prodtype']].head(1))
    
    show_images(df, index = indexes[:3], verbose = verbose)

    
def show_single_image_from_category(df, product_class, category, item, zoom = None, mark = None, verbose = False):
    
    ## if the ctagory name is passed
    if category not in product_class['prdtypecode'].to_list():
        category = product_class[product_class['prodtype'] == category]['prdtypecode'].values[0]

    ## directly here when the category code is passed
    indexes = df[ df['prdtypecode'] == category].index    
    idx = indexes[item]

    ## load image 
    file = "../datasets/image_train/image_"+str(df.loc[idx,'imageid'])+"_product_"+str(df.loc[idx,'productid'])+".jpg"
    image = np.int64(cv2.imread(file))
    print(file)

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    if zoom is not None:
        ax.imshow(image[zoom[0]:zoom[1],zoom[2]:zoom[3],::-1])
    else:
        ax.imshow(image[:,:,::-1])

    if mark is not None:
        ax.plot(mark[0]-5,mark[1]-5,marker = '*')
        ax.axhline(y = mark[0])
        ax.axvline(x = mark[0])
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.set()
    
    

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

    
    
def get_image_mean_RGB(df, image_type, Nb_pixels, verbose = True):
    '''
    takes a df with the vectorized images and get the mean R, G, B for each image.
    image_type = 'cropped_image' to indicate if images are reconstructed from the df or
    if they are fetched directly from the image folder (image_type = 'raw_image')
    '''
    t0 = time.time()
    checkpoints = [1000,2000,3000,4000]
    
    image_meanRGBs = np.empty((df.shape[0], 3)) 
    idx = 0

    # fetch raw image
    if image_type == 'raw':
        for idx in range(df.shape[0]):
            file = "./datasets/image_train/image_"+str(df.loc[idx,'imageid'])+"_product_"+str(df.loc[idx,'productid'])+".jpg"
            image = cv2.imread(file)        

            image_meanRGBs[idx, : ] = [ image[:,:,channel].mean() for channel in range(3) ]

            if verbose:
                if ((idx in checkpoints) or (idx%5000 == 0)):
                    print("%d images at time %0.2f minutes" %(idx, ((time.time()-t0)/60) ) )
    
    
    # reconstruc cropped image from passed datafrae
    if image_type == 'cropped':
        for idx in range(df.shape[0]):
            image = df.iloc[idx, 3:]#.to_numpy(copy = True)
            image = np.array(image).reshape(Nb_pixels, Nb_pixels, 3)

            image_meanRGBs[idx, : ] = [ image[:,:,channel].mean() for channel in range(3) ]

            if verbose:
                if ((idx in checkpoints) or (idx%5000 == 0)):
                    print("%d images at time %0.2f minutes" %(idx, ((time.time()-t0)/60) ) )

        
    t1 = time.time()
    if verbose:
        print("Getting mean RGB for %d images takes %0.2f minutes"%(df.shape[0],((t1-t0)/60)) )

    return image_meanRGBs

    
    
    