import pandas as pd
import numpy as np
from scipy.io import loadmat
import scipy.signal as signal

def feature_extraction(dataChTimeTr, Fs, included_features):
    channelFeatureTables = {}
    channel_index = 0
    for channel in dataChTimeTr:
        df = pd.DataFrame()
        for feature in included_features:
            if feature == "var":
                df[feature] = var(np.transpose(channel))
            elif feature == "waveformLength":
                df[feature] = waveformLength(np.transpose(channel))
            elif feature == "MAV":
                df[feature] = MAV(np.transpose(channel))
            elif feature == "RMS":
                df[feature] = RMS(np.transpose(channel))
            elif feature == "MNF":
                df[feature] = MNF(np.transpose(channel), Fs)
            elif feature == "frequencyRatio":
                df[feature] = MNF(np.transpose(channel), Fs)
        channelFeatureTables[channel_index] = df
        channel_index += 1

    return channelFeatureTables



def var(data):
    """Calculates the variance of signals in each trial of a given channel"""
    return np.var(data, axis = 1)

def waveformLength(data):
    """Calculates the waveform length of signals in each trial of a given channel"""
    return np.sum(np.abs(np.diff(data, axis = 1)), axis = 1)

def MAV(data):
    """Calculates the mean absolute value of signals in each trial of a given channel"""
    return np.sum(np.abs(data), axis = 1)

def RMS(data):
    """Calculates the Root Mean Square of signals in each trial of a given channel"""
    data = data ** 2
    square_sum = np.sum(data, axis = 1)
    return square_sum/data.shape[1]

def MNF(data, fs):
    """Calculates the Mean Frequencies of signals in each trial of a given channel"""
    MNFs = []
    window_size = 0.1
    overlap = 0.5
    nperseg = int(window_size * fs)
    noverlap = int(overlap * nperseg)

    for row in data:
        frequencies, power_spectrum = signal.welch(row, fs, nperseg = nperseg, noverlap = noverlap)
        product_sum = np.dot(frequencies, power_spectrum)
        intensity_sum = np.sum(power_spectrum)
        mnf = product_sum/intensity_sum
        MNFs.append(mnf)

    return np.transpose(np.array(MNFs))

def frequencyRatio(data, fs):
    """Calculates the frequency ratio of signals in each trail of a given channel"""
    FRs = []
    MNFs = MNF(data) # Using mean frequency as the threshold
    window_size = 0.1
    overlap = 0.5
    nperseg = int(window_size * fs)
    noverlap = int(overlap * nperseg)

    for row, mnf in data, MNFs:
        frequencies, power_spectrum = signal.welch(row, fs, nperseg=nperseg, noverlap=noverlap)
        highFrequency = np.where(frequencies > mnf, 1, 0)
        lowFrequency = np.where(frequencies <= mnf, 1, 0)
        highFrequency_sum = np.dot(power_spectrum, highFrequency)
        lowFrequency_sum = np.dot(power_spectrum, lowFrequency)
        fr = highFrequency_sum/lowFrequency_sum
        FRs.append(fr)

    return np.transpose(np.array(FRs))

if __name__ == '__main__':
    training_path = "exampleEMGdata180trial_train.mat"
    test_path = "exampleEMGdata120trial_test.mat"
    data = loadmat(training_path)
    print(data.keys())
    dataChTimetr = data['dataChTimeTr']
    print(np.transpose(dataChTimetr[1]).shape)
    included_features = ['var', 'waveformLength', 'MAV', 'RMS', 'MNF', 'frequencyRatio']
    print(feature_extraction(dataChTimetr, Fs = 1000, included_features = included_features)[0])
