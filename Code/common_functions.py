import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
import random
from IPython.display import clear_output
import shutil
import time
import os

def LoadData(image_size: tuple = (480, 640), seed: int = 1234, color: str = "rgb", shuffle: bool = True, batch_size: int = 32, path: str = "../Data/Original/ds1/") -> tuple:
    """
    Load all the images in the given dataset folder. Since all the images in the
    dataset are already split in separate folders, this function
    will extract each and return them as a tuple of `tf.data.Dataset`, along with
    a list of the class names.

    Args:
        image_size: size the images will be processed into (h,w) 
            (default = (480,640))
        seed: random seed to shuffle the dataset with. Use `None` if do not need reproducibility.
            (default = 1234)
        color: the color to process the images as. ("rgb", "rgba", "grayscale)
            (default = "rbg")
        shuffle: Whether or not to shuffle the dataset.
        batch_size: Default batch size for loading in the datasets, default is 32.
        path: the path to the folder containing the "ds" folders with the images. (make sure the path ends with a "/")

    Returns:
        `tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[string]]`
        where it is (Train, Test, Validation, and class names) respectively
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{path}Train/",
        image_size=image_size,
        color_mode=color,
        seed=seed,
        shuffle = shuffle,
        batch_size = batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{path}Test/",
        image_size=image_size,
        color_mode=color,
        seed=seed,
        shuffle = shuffle, 
        batch_size = batch_size
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        f"{path}Validation/",
        image_size=image_size,
        color_mode=color,
        seed=seed,
        shuffle = shuffle,
        batch_size = batch_size 
    )
    return (train_ds, test_ds, validation_ds, train_ds.class_names)

def PeakData(dataset: tf.data.Dataset,
             class_names: list,
             nrows: int = 3,
             ncols: int = 3,
             prediction_labels: list = None
            ) -> None:
    """
    Displays the images in the given dataset. If predictions are given, it will
    say in the title what the prediction was vs the what it actually is.

    Args:
        dataset: the dataset to view some images from
        class_names: list of the names of the classifications
        nrows: number of rows to display (default = 3)
        ncols: number of columns to display (default = 3)
        prediction_labels: list of the predictions. If None, it won't be used
            (default = None)
    """
    for images, labels in dataset.take(1):
        for i in range(nrows * ncols):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]] if prediction_labels == None else f"pred: {class_names[prediction_labels[i]]} | actual: {class_names[labels[i]]}")
            plt.axis("off")
    plt.show()

def ExtractLabels(dataset: tf.data.Dataset) -> list:
    """
    Take the given dataset and return a list of its labels. This can take some
    time, try to store results into a variable when you can.

    Args:
        dataset: the dataset to extract labels from
    
    Returns:
        `list[int]` - list of the labels in the dataset
    """
    return list(
        dataset.map(lambda _,y: y)
            .flat_map(tf.data.Dataset.from_tensor_slices)
            .as_numpy_iterator()
    )

def ConfusionMatrix(class_names: list, true_labels: list, predicted_labels: list) -> None:
    """
    Displays a confusion matrix for the predictions.

    Args:
        class_names: list of the classification names
        true_labels: labels of the dataset that was tested
        predicted_labels: list of the predictions made
    """
    # Create a confusion matrix as a 2D array.
    confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)

    # Use a heatmap plot to display it.
    ax = sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='.3g', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )

    # Add axis labels.
    ax.set(xlabel='Predicted Label', ylabel='True Label')
    plt.show()

def EvaluateModel(model: tf.keras.Sequential, test_ds: tf.data.Dataset, history: tf.keras.callbacks.History) -> None:
    """
    Take the model and plot the training accuracy and validation accuracy. Also,
    Perform a evaluation on the test data and print the loss and accuracy.
    Args:
        model: the model to test
        test_ds: the test dataset to evaluate the model with
        history: the history from fitting the model
    Returns:
        Tuple[float, float, float]
        (Train accuracy, Validation accuracy, Test accuracy)
    """
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()
    hist = history.history
    x_arr = np.arange(len(hist['loss'])) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
    ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
    ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")
    return hist['accuracy'][-1], hist['val_accuracy'][-1], test_acc

def MakePredictions(model: tf.keras.Sequential, test_ds: tf.data.Dataset) -> tuple:
    """
    Extract prediction and test vectors from tensorflow dataset
    Args:
        model: the model to test
        test_ds: the test dataset to evaluate the model with 
    """
    y_pred_probs = []
    y_test = []
    for batch in test_ds.as_numpy_iterator():
        x_batch, y_batch = batch
        y_pred_batch = model.predict(x_batch, verbose = 0)
        y_pred_probs.append(y_pred_batch)
        y_test.append(y_batch)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test = np.concatenate(y_test, axis=0)
    return (y_pred_probs, y_pred, y_test)
    
def PrecisionRecallScores (y_test:np.array, y_pred:np.array) -> None:
    """
    Calculate precision, recall, and F1 score on the test dataset.
    Micro averaged precision: calculate precision/recall for all classes, take average. 
    Treats all classes equally, gives idea of overall performance.
    Macro averaged precision: calculate class wise TP and FN, use to calculate overall precision and recall. 
    Better for evaluating the model's performance across each class separately. 
    Since our classes are balanced, we will use macro averaging.
    F1 score is the harmonic mean of precision and recall

    Args:
        y_test: the array of actual class values
        y_pred: the array of predicted class values
    """       
    
    #Calculate macro averaged precision and recall
    macro_averaged_precision = metrics.precision_score(y_test, y_pred, average = 'macro')
    macro_averaged_recall = metrics.recall_score(y_test, y_pred, average = 'macro')
    macro_averaged_f1score = metrics.f1_score(y_test, y_pred, average = 'macro')
    
    #Print results
    print(f"Macro averaged precision score: {macro_averaged_precision}")
    print(f"Macro averaged recall score: {macro_averaged_recall}")
    print(f"Macro averaged F1 score: {macro_averaged_f1score}")

def AugmentImage(brightness: float = 0.0, contrast: int = 1, flip: bool = False, hue: float = 0.0, gamma: int = 1, saturation: float = 0.0):
    def AugmentImageHelper(x, y):
        aug = x
        if hue != 0.0:
            aug = tf.image.adjust_hue(aug, hue)
        if brightness != 0.0:
            aug = tf.image.adjust_brightness(aug, delta=brightness)
        if gamma != 1:
            aug = tf.image.adjust_gamma(aug, gamma=gamma)
        if contrast != 1:
            aug = tf.image.adjust_contrast(aug, contrast_factor=contrast)
        if saturation != 0.0:
            aug = tf.image.adjust_saturation(aug, saturation_factor=saturation)
        if flip:
            aug = tf.image.random_flip_left_right(aug)
        return aug, y
    return AugmentImageHelper

def ROCPlots(y_pred_probs:np.array, y_test: np.array, y_pred: np.array, class_names: str) -> None:
    """
    Plots the ROC curve for each class, macro-averaged ROC curve
    Args:
        y_pred_probs: the array of class probabilities 
        y_test: the array of actual class values
        y_pred: the array of predicted class values
    """
    
    # Compute ROC curve and ROC area for each class
    n_classes = 13
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve((y_test==i).astype(int), y_pred_probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
     
    ##Plot the ROC curve for each class
    fig, axes  = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'orange', 'brown', 'pink', 'olive', 'purple', 'tomato']
    for i, color in zip(range(n_classes+1), colors):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        if i < 13:
            ax.plot(fpr[i], tpr[i], color=color, lw=2, label = "Model ROC curve")

           
        if i == 13:
             #Plot macro averaged curve
             ax.plot(fpr["macro"], tpr["macro"],
    label='Average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]), color='navy', linestyle='-', linewidth=2)

         # Plot the diagonal line representing the random classifier
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

        # Customize the plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(class_names[i]) if i < 13 else ax.set_title("Macro-average ROC curve")
        ax.legend(loc = "lower right")
    axes[3,2].set_visible(False)
    axes[3,3].set_visible(False)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()
    
def precision_recall_metrics(model: tf.keras.Sequential, test_ds: tf.data.Dataset, class_names:str) -> None:
    """
    Runs functions for calculating and visualizating precision and recall
    Args:
        model: the model to test
        test_ds: the test dataset to evaluate the model with 
    """
    y_pred_probs, y_test, y_pred = MakePredictions(model, test_ds)
    PrecisionRecallScores(y_test, y_pred)
    ConfusionMatrix(class_names, y_test, y_pred)
    ROCPlots(y_pred_probs, y_test, y_pred, class_names)

def reduce_dimensions_svd(dataset: tf.data.Dataset, k:int = 32) -> tf.data.Dataset:
    """
    Uses singular value decomposition to reduce the dimensions of the dataset
    Args:
        dataset: the dataset to reshape
        k: the number of dimensions to reduce the dataset to
    """
    reduced_dataset = []
    
    #Loop through dataset
    for batch in dataset.as_numpy_iterator():
        
        #Extract features and values
        x_batch, y_batch = batch 
        
        #Reshape batch
        x_batch = tf.reshape(x_batch, (batch.shape[0], -1))

        # Compute SVD
        s, U, V = tf.linalg.svd(x_batch, full_matrices=False)

        # Truncate SVD matrices to desired number of reduced dimensions
        U = U[:, :k]
        s = s[:, :k]
        V = V[:, :k]

        # Compute reduced batch
        reduced_batch = tf.linalg.matmul(U, tf.linalg.matmul(tf.linalg.diag(s), tf.linalg.matrix_transpose(V)))

        # Reshape reduced batch to original shape
        reduced_batch = tf.reshape(reduced_batch, batch.shape)

        # Append reduced batch to reduced dataset
        reduced_dataset.append((reduced_batch, y_batch))
       
    return tf.data.Dataset.from_tensor_slices(reduced_dataset)

def CNNModel(class_names: list, conv_layers: list = [32], layers: list = [], learning_rate: float = 0.001, dropout: float = 0.5) -> tf.keras.Sequential:
    """
    Simple straight forward CNN model. this is just for simplicity and testing
    atm. I will make it more modular later once I know what we are doing
    Args:
        class_names: list of the classification names
        conv_layers: list of how many filters each convolutional layer should use
        layers: list with the sizes of each hidden layer
        learning_rate: the learning rate for the optimizer
        dropout: the dropout rate for the model
    Returns:
        `tf.keras.Sequential` - a constructed tf model
    """
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255))
    for filter_count in conv_layers:
        model.add(tf.keras.layers.Conv2D(filter_count, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    for layer_count in layers:
        model.add(tf.keras.layers.Dense(layer_count, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dropout))
    model.add(tf.keras.layers.Dense(len(class_names), activation = 'softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model

def AugmentImages(src: str = "../Data/Original/ds1/", dest: str ="../Data/Augmented/", seed = None) -> None:
    """
    Copies the images in the `src` data folder into a new `dest` folder. The destination will automatically
    create "Train", "Test", and "Validation" folders including the classification folders. There will be
    2 copies of each image from the source dataset. 1 image will be original (no augments) the other image
    will have augments (augmented image names are prefix with "_aug"). The randomness is determined by the
    order of the list of files mod 4 (4 for each augment). essentially it is uniformily random. A seed can
    be used to get a consistant result (but im going to be honest i haven't tried it yet).

    Args:
        src: the source data folder to copy from (must be one of the "ds" folders) (must end with a "/")
        dest: the destination folder to create "Train", "Test", and "Validation" folders (including the
            classification folders) and copy the images from the source to. (must end with a "/")
        seed: can use a seed for more deterministic results. use `None` if you don't want to use a seed,
            then the randomness will be uniformily random by just using the file index mod 4.
    """
    if seed != None:
        random.seed(seed)
    if not os.path.exists(dest):
        os.mkdir(dest)

    print("Checking files if they need to be renamed to jpeg")
    for sub_folder in ["Train", "Test", "Validation"]:
        src_sub_path = src + sub_folder + "/"
        dest_sub_path = dest + sub_folder + "/"
        if not os.path.exists(dest_sub_path):
            os.mkdir(dest_sub_path)
        for img_folder in [p for p in os.listdir(src_sub_path) if not p.startswith(".")]:
            src_img_path = src_sub_path + img_folder + "/"
            dest_img_path = dest_sub_path + img_folder + "/"
            if not os.path.exists(dest_img_path):
                os.mkdir(dest_img_path)
            num_files = len(os.listdir(src_img_path))
            for i, file_name in enumerate(os.listdir(src_img_path)):
                if i % 100 == 0:
                    clear_output(wait=True)
                    print(f"{i} / {num_files} | {src_img_path} -> {dest_img_path}")
                    time.sleep(0.01)
                if file_name.endswith(".jpg"):
                    new_file_name = file_name.replace(".jpg", ".jpeg")
                    shutil.copy(os.path.join(src_img_path, file_name), os.path.join(dest_img_path, new_file_name))
                    shutil.copy(os.path.join(src_img_path, file_name), os.path.join(dest_img_path, new_file_name[:-5] + "_aug.jpeg"))

    clear_output(wait=True)
    print("Running augments")
    time.sleep(0.5)
    totals = {}
    folder_total = sum([len(os.listdir(dest + sub_folder + "/")) for s in ["Train", "Test", "Validation"]])
    folder_count = 0
    for sub_folder in ["Train", "Test", "Validation"]:
        totals[sub_folder] = {
            "original": 0,
            "augmented": 0,
            "running_total": 0
        }
        dest_sub_path = dest + sub_folder + "/"
        for img_folder in os.listdir(dest_sub_path):
            dest_img_path = dest_sub_path + img_folder + "/"
            dest_img_list = [p for p in os.listdir(dest_img_path) if p.endswith("_aug.jpeg")]
            num_files = len(dest_img_list)
            totals[sub_folder]["original"] += len(os.listdir(dest_img_path)) - num_files
            totals[sub_folder]["augmented"] += num_files
            totals[sub_folder]["running_total"] += totals[sub_folder]["augmented"] + totals[sub_folder]["original"]
            for i, file_name in enumerate(dest_img_list):
                if i % 100 == 0 or i == num_files:
                    clear_output(wait=True)
                    print(f"{i} / {num_files} | {dest_img_path} ({folder_count} / {folder_total})")
                data = tf.image.decode_jpeg(tf.io.read_file(dest_img_path + file_name))
                data = ([
                    lambda: tf.image.random_hue(data,0.5,seed),
                    lambda: tf.image.random_brightness(data,0.8,seed),
                    lambda: tf.image.random_contrast(data,0.2,0.8,seed),
                    lambda: tf.image.random_saturation(data,0.2,0.8,seed)
                ])[((random.randomint(0,4) if seed != None else 0) + i) % 4]()
                tf.keras.utils.save_img(dest_img_path + file_name[:-5] + ".jpeg", data)
            folder_count += 1
    totals["totals"] = {}
    totals["totals"]["original"] = sum([totals[t]["original"] for t in ["Train", "Test", "Validation"]])
    totals["totals"]["augmented"] = sum([totals[t]["augmented"] for t in ["Train", "Test", "Validation"]])
    totals["totals"]["running_total"] = sum([totals[t]["running_total"] for t in ["Train", "Test", "Validation"]])
    print("Augmented images finished!")
    print(totals["totals"])
