import numpy as np
import pandas as pd
import os 
import cv2
import time

################################################################################################################

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



#################################################################################################################


def crop_resize_images(df, threshold, new_pixel_nb, path, new_dir, verbose = False):
    
    if ('productid' not in df.columns) or ('imageid' not in df.columns):
        print("Image data cannot be found from information on the dataframe. Try with another dataset.")
        return None
        
    t0 = time.time()
    
    create_folder_preprocessed_images(new_dir)
    
    for i, idx in enumerate(df.index):
        
        # load image
        filename = "image_" + str(df.loc[idx,'imageid'])+"_product_" + str(df.loc[idx,'productid'])+".jpg"

        file = path + filename
        image = cv2.imread(file)
        
        # crop image 
        cropped_image = crop_image(image, threshold = threshold)
        
        # resize image (downscale)
        resized_image = cv2.resize(cropped_image, (new_pixel_nb, new_pixel_nb))
      
        # save array as a new image in newly created folder, same image name
        new_file = new_dir + '/' + filename
        cv2.imwrite( new_file, resized_image)
#         print(new_file)
    
        if verbose:
            checkpoints = [1000,2000,3000,4000]
            if (((i+1) in checkpoints) or (i+1)%5000 ==0):
                print("%d images at time %0.2f minutes" %(i+1, ((time.time()-t0)/60) ) )
    
    t1 = time.time()
    if verbose:
        #print("Vectorization of %d images takes %0.2f seconds" %(df.shape[0],(t1-t0)) )
        print("Crop, resize and save %d images takes %0.2f minutes" %(df.shape[0],((t1-t0)/60)) )                

    return
    
    
    
def create_folder_preprocessed_images(new_dir):
    
#     folder = new_path + new_folder
    
    if not os.path.exists(new_dir):
        print('created folder', new_dir)
        os.mkdir(new_dir)
    else:
        print( new_dir, 'already exists!')
    

    

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



####################################################################################################################

'''
Create the subfolders with the given labels from the data frame
'''
def create_folders(df_y, subset_path = './subset/'):
    
    categories = np.unique(df_y.prdtypecode.values)
    
    for i in categories:
        folder = subset_path + str(i)
        
        if not os.path.exists(folder):
            print('created folder', i)
            os.mkdir(folder)
        else:
            print( str(i), ' exists!')



'''
function to move the images to their corresponding folder
'''
def move_files(df_X, df_y, subset_path = './subset/', verbose = False):
    failed = 0
    
    for idx in df_X.index:
        
        filename = "image_" + str(df_X.loc[idx,'imageid']) + "_product_" + str(df_X.loc[idx,'productid'])+".jpg"
        file_path = subset_path + r"{}/".format( str(df_y.loc[idx,'prdtypecode']) )
        file = file_path + filename
        
#         filename = r"../input/train/{}/{}.jpg".format(row.landmark_id, row.id )
#         oldfile = r"../input/train/{}.jpg".format(row.id )
        oldfile = subset_path + filename
    
        if not os.path.exists(file):
            try:
                os.rename(oldfile, file)
                print('moved {} to {}'.format(filename, file_path)) if verbose else None
            except:
                failed +=1
        else:
            print('{} is in {}'.format(filename, file_path)) if verbose else None
    
    print('failed on {} files'.format(failed))



'''
For the test folder, we need to create a dummy subfolder. In this case, we created 0
'''
def create_test_subfolder( subset_path = './subset/', verbose= False):
    count = 0
    
#     folder = r"../input/test"    
    folder = subset_path + "0"    
    
    if not os.path.exists(folder):
            os.mkdir(folder)
    
    for root, dirs, files in os.walk(subset_path):
        if '0' in dirs:
            for f in files:
                oldfile  = os.path.join(root, f)    
                newfile = '{}0/{}'.format(root,f)
                os.rename(oldfile,newfile)

                count +=1
                if verbose:
                    if count % 5000 == 0:
                        print('moved {} images, such as {} to {}'.format(count, oldfile,newfile))
                        
    print("moved {} files to {}".format(count, folder))








