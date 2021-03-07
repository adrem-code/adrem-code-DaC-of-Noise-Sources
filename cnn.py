import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score, classification_report
import itertools
from itertools import cycle
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.datasets import make_classification

DATA_PATH = "C:\\AED\\MEL_single.json"#datapath of json file

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["parameters"])
    y = np.array(data["labels"])
    Classes = np.array(data["mapping"])
    
    return X, y, Classes

def prepare_datasets(test_size, validation_size):
    # load data
    X, y,_ = load_data(DATA_PATH)
    
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = 42)
    
    #create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size,random_state = 42)
    # 3d array -> (no_of_timebins, no_of_mfcc_values, 1)
    X_train = X_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, y

def bulid_model(input_shape, no_of_categories):

    activ = 'tanh'
 
    #create model
    model = keras.Sequential()

    #1st conv layer
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(5,5), input_shape=input_shape,padding='same',activation=activ))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    


    # #2nd conv layer
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(5,5), input_shape=input_shape,padding='same',activation=activ))
    model.add(keras.layers.MaxPool2D(pool_size=(2,1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    
    

    # # # #3rd conv layer
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(5,5), input_shape=input_shape,padding='same', activation=activ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    
    

    #flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation=activ))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(no_of_categories, activation='softmax'))

    return model

def predict(model, X, y):
    
    X = X[np.newaxis, ...]
    #prediction is 2D array [[0.1, 0.2, ...], [...], ...] for every genre prediction
    prediction = model.predict(X) # X -> (1, 130, 13, 1)
    #extract index with max value
    predicted_index = np.argmax(prediction, axis=1) #1D array, returnes predicted index for specific category
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

def plot_history(history):
    
    fig, axs = plt.subplots(2)

    #create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("accuracy eval")

    #create error subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoc")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, y = prepare_datasets(0.25, 0.2)


    #bulid the CNN net
    input_shape = [X_train.shape[1], X_train.shape[2], X_train.shape[3]]
    print(input_shape)
    model = bulid_model(input_shape, len(list(set(y))))
    model.summary()
    #compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    
    #train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    #evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    #make prediction on a sample
    X=X_test[1]
    y=y_test[1]

    predict(model, X, y)

    predictions = np.argmax((model.predict(X_test) > 0.60).astype("int32"), axis=-1)

    _,_,cm_plot_labels = load_data(DATA_PATH)

    cm = confusion_matrix(y_true=y_test, y_pred=predictions)
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

    #plot accuracy and error over the epochs
    plot_history(history)

    #plot accuracy and error over the epochs
    plot_history(history)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


print(y_test.shape)
y_test_org = y_test
y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
print(y_test.shape)
y_score = model.predict(X_test)

print(y_score.shape)


n_classes=4

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


y_prob = model.predict(X_test)

macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                  average="macro")
weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                     average="weighted")
macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                  average="macro")
weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                     average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

print(y_test_org.shape)
print(y_test_org)
y_score = np.argmax((model.predict(X_test) > 0.60).astype("int32"), axis=-1)
print(y_score.shape)
print(y_score)
print(classification_report(y_test_org, y_score, target_names=cm_plot_labels))