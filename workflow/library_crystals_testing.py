def segmenter_testing(normalized_folder, cl_filename, label_folder):
    '''
    this function applies an object segmenter to the testing data, segments normalized images, excludes labels with area < 50 pixels, creates a label folder and saves there the segmentation results;
    results displayed like: normalized image (left), segmented image (middle), size excluded segmented image (right)
    '''
    
    import apoc
    import os
    from skimage.io import imread, imsave
    import pyclesperanto_prototype as cle
    import matplotlib.pyplot as plt
    import numpy 
    
    file_list = sorted(os.listdir(normalized_folder))
    
    # show all images
    fig, axs = plt.subplots(1, 13, figsize=(30,30))
    for i, filename in enumerate(file_list):
        image = imread(normalized_folder + filename)
        cle.imshow(image, plot=axs[i])
    plt.show()
    
    #prediction
    os.makedirs(label_folder, exist_ok=True)
    segmenter = apoc.ObjectSegmenter(opencl_filename=cl_filename)
    
    # show all images
    for i, filename in enumerate(file_list):
        fig, axs = plt.subplots(1, 3, figsize=(25,25))
    
        #show original
        print(filename)
        image = imread(normalized_folder + filename)
        cle.imshow(image, plot=axs[0])
    
         #show result of object segmenter
        labels = segmenter.predict(image)
        cle.imshow(labels, plot=axs[1], labels=True)
    
        #show result of object segmenter with labels in range
        labels_in_range = cle.exclude_small_labels(source = labels, maximum_size = 50)
        cle.imshow(labels_in_range, plot=axs[2], labels = True)
    
        #show all images
        plt.show()
    
        #save result of object segmenter with labels in range
        imsave(label_folder + filename, labels_in_range)
        
def column_selection_testing(table):
    '''
    This function selects measurements from table and creates tables with different combination of size, shape and intensity measurements.
    It saves them as csv-files
    '''
    import pandas as pd
    
    # correlation-filtered table with intensity, shape and size measurements
    keep = table[['label', 'aspect_ratio', 'max_intensity', 'min_intensity', 'perimeter_skimage', 'area', 'mean_intensity', 'major_axis_length', 'minor_axis_length', 'circularity', 'solidity', 'eccentricity', 'roundness_skimage', 'median', 'sum', 'variance', 'perimeter_on_border','perimeter_on_border_ratio','filename']]
    keep.to_csv('testing_size_shape_intensity.csv')
    
    #intensity table
    df_intensity = keep[['label','max_intensity', 'mean_intensity', 'min_intensity','median', 'sum', 'variance', 'filename']]
    df_intensity.to_csv('testing_intensity.csv')
    
    #size table
    df_size = keep[['label','area','filename']]
    df_size.to_csv('testing_size.csv')
    
    #shape table
    df_shape = keep[['label', 'aspect_ratio','perimeter_skimage', 'major_axis_length',
       'minor_axis_length', 'circularity', 'solidity', 'eccentricity',
       'roundness_skimage', 'perimeter_on_border',
       'perimeter_on_border_ratio', 'filename']]
    df_shape.to_csv('testing_shape.csv')
    
    #size intensity table
    df_size_intensity = pd.merge(df_size, df_intensity, on=('label','filename'))
    df_size_intensity_ordered = df_size_intensity.iloc[0:,[0, 1, 3, 4, 5, 6, 7, 8, 2]]
    df_size_intensity_ordered.to_csv('testing_size_intensity.csv')
    
    #size shape table
    df_size_shape = pd.merge(df_size, df_shape, on=('label','filename'))
    df_size_shape_ordered = df_size_shape.iloc[0:,[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 2]]
    df_size_shape_ordered.to_csv('testing_size_shape.csv')
    
    #df_shape_intensity = pd.merge(df_shape, df_intensity, on=('label','filename'))
    df_shape_intensity = pd.merge(df_shape, df_intensity, on=('label','filename'))
    df_shape_intensity_ordered = df_shape_intensity.iloc[0:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,11]]
    df_shape_intensity_ordered.to_csv('testing_shape_intensity.csv')
    


def classifier_testing(filtered_table, label_folder, tabrowcl_filename, classifier_label_folder, good_crystal_table_filename):
    '''
    this function applies the classifier to the testing data.
    it creates:
    - an output folder
    - writes the good crystal labels to a new table
    '''
    
    #import statements
    from skimage.io import imread, imsave
    from pyclesperanto_prototype import imshow, replace_intensities
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import apoc
    import os
    import pyclesperanto_prototype as cle
    from skimage.segmentation import relabel_sequential
    from matplotlib.colors import LinearSegmentedColormap
    
    #sort
    file_list = sorted(os.listdir(label_folder))
    table = pd.read_csv(filtered_table)
    
    label_list = []

    for i, filename in enumerate(file_list):
        #read label, annotation
        label = imread(label_folder + filename)

        #append label and annotation to list
        label_list.append(label)
    
    #prediction
    classifier = apoc.TableRowClassifier(opencl_filename= tabrowcl_filename, max_depth=2, num_ensembles=10)
    predicted_classes = classifier.predict(table)

    #append predicted classes as columns to the table
    table['predicted_class'] = predicted_classes
    
    #create list with number of objects in label_image
    number_elements = []
    for label in label_list:
        in_label = np.unique(label)
        number_elements.append(max(in_label))
    
    #slice the annotation and prediction list
    def make_slices(list_of_sizes):
        '''
        Make a list of slice objects from a list of sizes
        '''
        # sizes = [len(n) for n in list_of_objects] 
        slice_list = []
        for i, n in enumerate(list_of_sizes):
            if i==0:
                n_1 = 0
                slc = slice(n)
            else:        
                slc = slice(n_1, n_1 + n)
            n_1 += n
            slice_list += [slc]
        return slice_list
    
    slice_list = make_slices(number_elements)
    
    prediction_per_image_list = []

    for slc in slice_list:
        # get prediction for every single image
        prediction_per_image = table['predicted_class'] [slc]
        prediction_per_image_list.append(prediction_per_image)
    
    #add background to the prediction
    predicted_classes_with_background = []
    prediction = []
    for prediction in prediction_per_image_list:
        predicted_class_with_background = [0] + prediction.tolist()
        predicted_classes_with_background.append(predicted_class_with_background)
        
    #colormap
    my_colors = [
        [0,0,0,1],
        [1,0,1,1],
        [0,1,0,1]
    ]
    colormap = LinearSegmentedColormap.from_list("green_magenta", my_colors)
    
    #create folder for classifier results 
    os.makedirs(classifier_label_folder, exist_ok=True)
    for label, prediction,filename in zip(label_list,predicted_classes_with_background,file_list):
        print(filename)
    
        # connect prediction to label image
        class_image = replace_intensities(label, prediction).astype(int)
    
        #show prediction
        imshow(class_image, colorbar= True,colormap= colormap,min_display_intensity=0, max_display_intensity = 2)
    
        #save prediction
        imsave(classifier_label_folder+filename, class_image)
    
    #save good crystal table
    good_crystals = table.loc[table['predicted_class'] == 2]
    good_crystals.to_csv(good_crystal_table_filename)