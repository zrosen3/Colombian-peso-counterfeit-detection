import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def LoadData(image_size: tuple = (480, 640), seed: int = 1234, ds_num: int = 1, color: str = "rgb", shuffle = True, batch_size = 32) -> tuple:
    """
    Load all the images in the given dataset folder. Since all the images in the
    dataset are already split in separate folders, this function
    will extract each and return them as a tuple of `tf.data.Dataset`, along with
    a list of the class names.

    Args:
        image_size: size the images will be processed into (h,w) 
            (default = (480,640))
        seed: random seed to shuffle the dataset with. Use `None` if do not need reproducibility
        . (default = 1234)
        ds_num: the dataset number corrisponding to the folder to extract the
            dataset from. (ex. 1 = "ds1") (default = 1)
        color: the color to process the images as. ("rgb", "rgba", "grayscale)
            (default = "rbg")
        shuffle: Whether or not to shuffle the dataset.
        batch_size: Default batch size for loading in the datasets, default is 32.
    Returns:
        `tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[string]]`
        where it is (Train, Test, Validation, and class names) respectively
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"../Data/Original/ds{ds_num}/Train/",
        image_size=image_size,
        color_mode=color,
        seed=seed,
        shuffle = shuffle,
        batch_size = batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"../Data/Original/ds{ds_num}/Test/",
        image_size=image_size,
        color_mode=color,
        seed=seed,
        shuffle = shuffle, 
        batch_size = batch_size
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        f"../Data/Original/ds{ds_num}/Validation/",
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
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")