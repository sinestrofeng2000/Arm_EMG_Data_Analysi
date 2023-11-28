from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from Feature_Extraction import feature_extraction
import matplotlib.pyplot as plt

def SingleFeatureVectors(featureTables):
    """
    Docstring: Given a dictionary of feature tables in each of the 4 channels, this function outputs a 3d numpy array containing
    extracted features from all trials in all 4 channels.

    Parameters:
    - parameter1 (featureTables): A dictionary with 4 items, each corresponding to the feature table extracted from signals
    in a single channel

    Returns: Features extracted from all trials and all channels
    Return type: 3d numpy array
    """
    channel1 = featureTables[0].values
    channel2 = featureTables[1].values
    channel3 = featureTables[2].values
    channel4 = featureTables[3].values
    SingleFeatureVectors = np.concatenate((channel1[:, :, None], channel2[:, :, None], channel3[:, :, None], channel4[:, :, None]), axis = 2)
    return SingleFeatureVectors

def ConfMatForFeaturePairs(includedFeatures, singleFeatureVectors, labels):
    """
    Docstring: Plots and saves the confusion matrices of both the LDA and KNN classifiers trained on all possible pairs
    of given features

    Parameters:
    - parameter1 (includedFeatures): A list containing all feature names
    - parameter2 (singleFeatureVectors): A 3D numpy array containing all 6 features extracted from all 4 channels of all
    trials
    - parameter3 (labels): A numpy array containing gold labels for all 180 trials

    Returns: None
    Return type: None
    """
    for i in range(len(included_features)):
        for j in range(i + 1, len(included_features)):
            feature1 = np.squeeze(singleFeatureVectors[:, i, :])
            feature2 = np.squeeze(singleFeatureVectors[:, j, :])
            x_train = np.concatenate((feature1, feature2), axis = 1)
            y_train = labels
            KNN = KNeighborsClassifier(n_neighbors=3)
            LDA = LinearDiscriminantAnalysis()
            cv_predictions_knn = cross_val_predict(KNN, x_train, y_train, cv=5)
            cv_predictions_lda = cross_val_predict(LDA, x_train, y_train, cv=5)
            conf_matrix_knn = confusion_matrix(y_train, cv_predictions_knn)
            conf_matrix_lda = confusion_matrix(y_train, cv_predictions_lda)
            classes = ['Rock', 'Paper', 'Scissors']

            # Plot Confusion matrix for KNN
            display_knn = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_knn, display_labels=classes)
            display_knn.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation='vertical')
            plt.title('KNN for {}, {}'.format(includedFeatures[i], includedFeatures[j]))
            plt.show()

            # Plot Confusion matrix for LDA
            display_lda = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lda, display_labels=classes)
            display_lda.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation='vertical')
            plt.title('LDA for {}, {}'.format(includedFeatures[i], includedFeatures[j]))
            plt.show()




if __name__ == '__main__':
    # Read and load the data
    training_path = "exampleEMGdata180trial_train.mat"
    test_path = "exampleEMGdata120trial_test.mat"
    data = loadmat(training_path)
    labels = np.squeeze(data['labels'])
    dataChTimetr = data['dataChTimeTr']

    # Create a list containing names of all features to be tested
    included_features = ['var', 'waveformLength', 'MAV', 'RMS', 'MNF', 'frequencyRatio']

    # Generate a dictionary of feature tables in each of the 4 channels with the feature extraction function
    featureTables = feature_extraction(dataChTimetr, Fs = 1000, included_features = included_features)

    # Store feature tables in all 4 channels to a 3d numpy array
    singleFeatureVectors = SingleFeatureVectors(featureTables)

    # Standard scaler of feature values for the KNN classifier training
    scaler = StandardScaler()

    # Train KNN & LDA models on each feature in all 4 channels and store the cross-validation accuracy of each model in a list
    KNNAccuracies = []
    LDAAccuracies = []
    for i in range(len(included_features)):
        x_train = np.squeeze(singleFeatureVectors[:, i, :])
        x_train = scaler.fit_transform(x_train)
        y_train = labels
        KNN = KNeighborsClassifier(n_neighbors = 3)
        accuracy_KNN = cross_val_score(KNN, x_train, y_train, cv=5, scoring='accuracy')
        LDA = LinearDiscriminantAnalysis()
        accuracy_LDA = cross_val_score(LDA, x_train, y_train, cv=5, scoring='accuracy')
        KNNAccuracies.append(sum(accuracy_KNN)/5)
        LDAAccuracies.append(sum(accuracy_LDA)/5)

    # Bar width
    bar_width = 0.35

    indices = np.arange(6)

    # Create a bar plot
    plt.bar(indices, KNNAccuracies, bar_width, label='KNN', color='blue', alpha=0.7)
    plt.bar(indices + bar_width, LDAAccuracies, bar_width, label='LDA', color='orange', alpha=0.7)

    # Labeling and customization
    plt.xlabel('Feature')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison for KNN and LDA')
    plt.xticks(indices + bar_width / 2, included_features)
    plt.legend()

    # Show the plot
    plt.show()

    # Plot confusion matrices for KNN/LDA classifiers trained on all pairs of features
    ConfMatForFeaturePairs(included_features, singleFeatureVectors, labels)






