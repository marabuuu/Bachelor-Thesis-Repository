def data_selection(filename, image_folder, viewer):
    '''
    this function loads one image slice by slice in napari
    '''
    
    from aicsimageio import AICSImage
    from aicsimageio.readers import BioformatsReader
    
    #load image
    image = AICSImage(image_folder + filename, reader = BioformatsReader)
    original = image.data
    
    #select reflection channel
    channel2 = original[0,2]
    
    #add every slice of the stack to napari
    for slc in range(channel2.shape[0]):
        layer2 = viewer.add_image(channel2[slc], colormap = 'cyan', blending = 'translucent')
        
        
def normalization(input_folder, normalized_folder):
    '''
    this function creates an output folder and normalizes the input images by dividing the intensity of all pixels by the 99th percentile
    '''
    
    # import statements
    from skimage.io import imread, imsave
    import matplotlib.pyplot as plt
    import os
    import pyclesperanto_prototype as cle
    import numpy 

    #create output folder
    os.makedirs(normalized_folder, exist_ok=True)
    file_list = sorted(os.listdir(input_folder))
    
    fig, axs = plt.subplots(2, 13, figsize=(20,10))
    for i, filename in enumerate(file_list):
        #show original
        image = imread(input_folder + filename)
        cle.imshow(image, plot=axs[0,i])
    
        #show normalized
        normalized_image = image/numpy.percentile(image,99)
        cle.imshow(normalized_image, plot=axs[1,i], max_display_intensity=1)
    
        #save normalized
        imsave(normalized_folder + filename, normalized_image)
    plt.show()

    
def segmenter(normalized_folder,masks_folder, cl_filename, label_folder):
    '''
    this function trains an object segmenter, segments normalized images, excludes labels with area < 50 pixels, creates a label folder and saves there the segmentation results;
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

    # show corresponding label images
    fig, axs = plt.subplots(1, 13, figsize=(30,30))
    for i, filename in enumerate(file_list):
        masks = imread(masks_folder + filename)
        cle.imshow(masks, plot=axs[i])
    plt.show()
    
    # training 
    # setup classifer and where it should be saved
    apoc.erase_classifier(cl_filename)
    segmenter = apoc.ObjectSegmenter(opencl_filename=cl_filename)

    # setup feature set used for training
    features = "gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 sobel_of_gaussian_blur=1"

    # train classifier on folders
    apoc.train_classifier_from_image_folders(
        segmenter, 
        features, 
        image = normalized_folder, 
        ground_truth = masks_folder)
    
    #prediction
    os.makedirs(label_folder, exist_ok=True)
    
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

        
def table_generation(normalized_folder, label_folder, table):
    '''
    this function creates a table with parameters using skimage and simpleitk for every crystal label in a folder;
    the images are ordered underneath each other;
    the table is saved as a csv-file
    '''
    
    #import statements
    import os
    from skimage.io import imread, imsave
    import matplotlib.pyplot as plt
    import numpy as np 
    import pandas as pd
    from napari_skimage_regionprops import regionprops_table
    from napari_simpleitk_image_processing import label_statistics
    
    #sort files
    file_list = sorted(os.listdir(normalized_folder))
    
    #measure in normalized image
    def analyze_image(filename):
        image = imread(normalized_folder + filename)
        label = imread(label_folder + filename)
        
        # measure in normalized image
        df_skimage = pd.DataFrame(regionprops_table(image , label, size = True, intensity = True, perimeter = True, shape = True))
        df_skimage["aspect_ratio"] = df_skimage["major_axis_length"]/df_skimage["minor_axis_length"]
        df_simpleitk = pd.DataFrame(label_statistics(image, label, size = True, intensity = True, perimeter = True, shape = True))
        DF = pd.merge(df_skimage, df_simpleitk, on = "label", suffixes = ('_skimage', '_simpleitk'))
    
        #append filename
        DF["filename"] = filename
        to_drop = ["extent", "local_centroid-0", "local_centroid-1","orientation"]
        orientation_filtered = DF.drop(to_drop, axis=1)
        return orientation_filtered
    
    #append all images to the table
    df_all_images = pd.DataFrame()
    for i, filename in enumerate(file_list):
        df_one_image = analyze_image(filename)
        df_all_images = pd.concat([df_all_images, df_one_image], ignore_index = True)
    df_all_images.to_csv(table)
    return df_all_images


def correlation_filter(table, correlation_filtered_table):
    '''
    this function eliminates parameters that are correlating > 0.95 and only leaves one representative column inside;
    table = filename
    first, it prints out the parameters it drops;
    second, it prints out the parameters that are left in
    '''
    
    #import statements
    import numpy as np
    import pandas as pd
    
    #read in table
    table = pd.read_csv(table)
    
    #ask for measurements that are correlating > 0.95 
    cor_matrix = table.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print(); print(to_drop)

    #drop measurements that are correlating > 0.95
    correlation_filtered = table.drop(to_drop, axis=1)
    print(correlation_filtered.columns)
    correlation_filtered.to_csv(correlation_filtered_table)
    return correlation_filtered


def column_selection(table):
    '''
    This function selects measurements from table and creates tables with different combination of size, shape and intensity measurements.
    It saves them as csv-files
    '''
    
    # correlation-filtered table with intensity, shape and size measurements
    keep = table[['label', 'aspect_ratio', 'max_intensity', 'min_intensity', 'perimeter_skimage', 'area', 'mean_intensity', 'major_axis_length', 'minor_axis_length', 'circularity', 'solidity', 'eccentricity', 'roundness_skimage', 'median', 'sum', 'variance', 'perimeter_on_border','perimeter_on_border_ratio','filename']]
    keep.to_csv('size_shape_intensity.csv')
    
    #intensity table
    df_intensity = keep[['label','max_intensity', 'mean_intensity', 'min_intensity','median', 'sum', 'variance', 'filename']]
    df_intensity.to_csv('intensity.csv')
    
    #size table
    df_size = keep[['label','area','filename']]
    df_size.to_csv('size.csv')
    
    #shape table
    df_shape = keep[['label', 'aspect_ratio','perimeter_skimage', 'major_axis_length',
       'minor_axis_length', 'circularity', 'solidity', 'eccentricity',
       'roundness_skimage', 'perimeter_on_border',
       'perimeter_on_border_ratio', 'filename']]
    df_shape.to_csv('shape.csv')
    
    #size intensity table
    df_size_intensity = pd.merge(df_size, df_intensity, on=('label','filename'))
    df_size_intensity_ordered = df_size_intensity.iloc[0:,[0, 1, 3, 4, 5, 6, 7, 8, 2]]
    df_size_intensity_ordered.to_csv('size_intensity.csv')
    
    #size shape table
    df_size_shape = pd.merge(df_size, df_shape, on=('label','filename'))
    df_size_shape_ordered = df_size_shape.iloc[0:,[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 2]]
    df_size_shape_ordered.to_csv('size_shape.csv')
    
    #df_shape_intensity = pd.merge(df_shape, df_intensity, on=('label','filename'))
    df_shape_intensity = pd.merge(df_shape, df_intensity, on=('label','filename'))
    df_shape_intensity_ordered = df_shape_intensity.iloc[0:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,11]]
    df_shape_intensity_ordered.to_csv('shape_intensity.csv')



def classifier(filtered_table, label_folder, class_masks_folder, tabrowcl_filename, classifier_label_folder, good_crystal_table_filename):
    '''
    this function trains and predicts a classifier into good and bad crystal labels.
    it creates:
    - an output folder
    - a confusion matrix
    - computes accuracy, precision, recall, f-score
    - writes the good crystal labels to a new table
    - shows feature importance of the classifier
    '''
    
    #import statements
    from skimage.io import imread, imsave
    from pyclesperanto_prototype import imshow, replace_intensities
    from skimage.measure import label, regionprops
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import apoc
    import os
    import pyclesperanto_prototype as cle
    from skimage.segmentation import relabel_sequential
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    from sklearn import metrics
    from IPython.display import display
    from matplotlib.colors import LinearSegmentedColormap
    
    #sort
    file_list = sorted(os.listdir(label_folder))
    table = pd.read_csv(filtered_table)
    
    label_list = []
    annotation_list = []
    all_annotated_classes = []

    for i, filename in enumerate(file_list):
        #read label, annotation
        label = imread(label_folder + filename)
        annotation = imread(class_masks_folder + filename)

        #get all annotated classes
        annotation_stats = regionprops(label, intensity_image=annotation)
        annotated_classes = np.asarray([s.max_intensity for s in annotation_stats])
        all_annotated_classes = np.concatenate((all_annotated_classes, annotated_classes))

        #append label and annotation to list
        label_list.append(label)
        annotation_list.append(annotation)
    
    #training
    classifier = apoc.TableRowClassifier(opencl_filename= tabrowcl_filename, max_depth=2, num_ensembles=10)
    classifier.train(table.iloc[:,2:-1], all_annotated_classes)
    
    #prediction
    predicted_classes = classifier.predict(table)

    #append annotated and predicted classes as columns to the table
    table['annotated_class'] = all_annotated_classes
    table['predicted_class'] = predicted_classes
    
    #confusion matrix
    y_actual = table['annotated_class']
    y_predicted = table['predicted_class']
    
    all_annotated_classes_mask = all_annotated_classes > 0
    y_actual_without_background = all_annotated_classes[all_annotated_classes_mask]
    y_predicted_without_background = predicted_classes[all_annotated_classes_mask]
    confusion_matrix = metrics.confusion_matrix(y_actual_without_background, y_predicted_without_background)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    
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
    
    annotation_per_image_list =[]
    prediction_per_image_list = []

    for slc in slice_list:
        # get annotation for every single image
        annotation_per_image = table['annotated_class'][slc]
        annotation_per_image_list.append(annotation_per_image)
    
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
    
    #focus on objects that are annotated
    all_annotated_classes_mask = all_annotated_classes > 0

    #accuracy 
    print('Accuracy: %.3f' % accuracy_score(y_actual[all_annotated_classes_mask], y_predicted[all_annotated_classes_mask]))

    #precision
    print('Precision: %.3f' % precision_score(y_actual[all_annotated_classes_mask],y_predicted[all_annotated_classes_mask]))

    #recall
    print('Recall: %.3f' % recall_score(y_actual[all_annotated_classes_mask], y_predicted[all_annotated_classes_mask]))

    #F1-score
    print('F1 Score: %.3f' % f1_score(y_actual[all_annotated_classes_mask], y_predicted[all_annotated_classes_mask]))

    #feature importance
    def colorize(styler):
        styler.background_gradient(axis=None, cmap="plasma")
        return styler

    shares, counts = classifier.statistics()
    df = pd.DataFrame(shares).T
    df_beautiful = df.style.pipe(colorize)
    display(df_beautiful)
    
    feature_importances = classifier.feature_importances()
    df_feature_importances = pd.DataFrame(list(feature_importances.items()),columns = ['parameter','feature importance']) 
    df_colorful_feature_importance = df_feature_importances.style.pipe(colorize)
    display(df_colorful_feature_importance)
    
    #save good crystal table
    good_crystals = table.loc[table['predicted_class'] == 2]
    good_crystals.to_csv(good_crystal_table_filename)
    
    #number of objects in good crystal labels
    trends_table_good_crystals = good_crystals.describe()
    counted_objects_good_crystals =  trends_table_good_crystals.loc['count']['predicted_class']
    print(counted_objects_good_crystals)

def bad_label_exclusion(label_folder, classification_result_folder, good_crystal_label_folder):
    '''
    This function takes a deletes the bad labels from the label image and stores the good label images in a folder.
    They can be used to perform measurements on the good labels.
    '''
    
    #import statements
    import os
    from skimage.io import imread, imsave
    import numpy as np
    from pyclesperanto_prototype import imshow
    
    #sort and load images, labels, classification results
    file_list = sorted(os.listdir(label_folder))
    label_list = []
    classification_result_list = []

    for i, filename in enumerate(file_list):
        label = imread(label_folder + filename)
        classification = imread(classification_result_folder + filename)
        label_list.append(label)
        classification_result_list.append(classification)
    
    #create directory 
    os.makedirs(good_crystal_label_folder, exist_ok=True)
    
    for class_image,label,filename in zip(classification_result_list,label_list,file_list):
        #change class of bad crystals (1) to background (0)
        class_image = np.asarray(class_image)
        class_image[class_image == 1]=0
        class_image_mask = class_image.astype(bool)
    
        #exclude bad labels from label image
        label_image_filtered=np.copy(label)
        label_image_filtered[class_image_mask==False]=0
        print(filename)
    
        #show good labels 
        imshow(label_image_filtered, colorbar= True,colormap='jet',min_display_intensity=0)
    
        #save good labels
        imsave(good_crystal_label_folder+filename,label_image_filtered)