import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import librosa, librosa.display
import json
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling1D
import CepstralCoefficients

DATA_PATH = "C:\\AED\\MEL_single.json"
file_name = r"D:\Praca Inzynierska\Nagrania\Wierzynskiego_1\zrodla_sekwencja\5_12-36-55_12-37-42_ciezarowka_pociag.wav" #path of test audio
SAMPLE_RATE = 22050
win_len = 0.1
batch_size  = 5

n_fft = 2048
hop_length = 512
type_of_parameters = 'mel'
method_of_windowing = 'single'

n_ceps = 14
n_filter = 128
if type_of_parameters == 'mel' or type_of_parameters == 'erb':
    n_vectors = n_filter
if type_of_parameters == 'mfcc' or type_of_parameters == 'hfcc':
    n_vectors = n_ceps


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    Classes = np.array(data["mapping"])
    return Classes

def bulid_multi_instance(base, batches=5, n_vec=14, frames=5,  channels=1):
    '''
    n_vec is number of vectors of mel/erb/mfcc/hfcc
    frames is number of frames in sample
    batches number of averaged samples
    '''
    input = Input(shape=(batches, n_vec, frames,  channels))

    x= input
    x= TimeDistributed(base)(x)
    x= GlobalAveragePooling1D()(x)
    model = Model(input,x)
    return model

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


#bulid multi instance
model = bulid_multi_instance(loaded_model, batches=batch_size, n_vec = n_vectors, frames = 5)
#load audio file
signal, sample_rate = librosa.load(file_name, sr=SAMPLE_RATE)

Classes = load_data(DATA_PATH)

#create live plot
plt.ion()
fig,ax = plt.subplots()
plt.show()
bars = ax.barh(Classes,np.zeros(len(Classes)), align='center')

# process all segments of audio file
for d in range(int(len(signal)/SAMPLE_RATE/win_len/batch_size)):

    # calculate start and finish sample for current segment
    start = int(sample_rate * win_len * d * batch_size)
    finish = int(start + sample_rate*win_len)
    diff = (finish-start)
    n_batch = []
    if finish<=len(signal):
        for b in range(batch_size):
            # extract mfcc
            #RMS normalization
            signal_norm = CepstralCoefficients.RMSnormalization(signal[start+b*diff:start+(b+1)*diff], -1)
            #for mel spectrogram
            if type_of_parameters == 'mel':
                _, parameter = CepstralCoefficients.mfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
            #for erb spectrogram
            if type_of_parameters == 'erb':
                _, parameter = CepstralCoefficients.hfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
            #for mfcc
            if type_of_parameters == 'mfcc':
                parameter,_ = CepstralCoefficients.mfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
            #for hfcc
            if type_of_parameters == 'hfcc':
                parameter,_ = CepstralCoefficients.hfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
            n_batch.append(parameter.tolist())

    parameters = np.array(n_batch)
    X = parameters[np.newaxis,...,np.newaxis]
    prediction = model.predict(X)

    predicted_index = np.argmax(prediction, axis=1) #1D array, returnes predicted index for specific category
    print("Predicted index: {}".format(predicted_index))
    print("Assurance: "+'{:.2f}'.format(float(prediction[0][predicted_index]*100)))
    

    threshold = 60

    if float(prediction[0][predicted_index]*100) > threshold:
        textstr = str(Classes[predicted_index])
        textstr = textstr[2:-2]
    else:
        textstr = "Unknown"
    plt.cla()
    ax.barh(Classes, prediction[0]*100, color="blue")
    ax.plot( [threshold, threshold],[-0.5, 3.5], color="black")
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
    ax.set_xlim(0,100)
    ax.set_ylim(-0.5,3.5)    
    plt.draw()
    plt.pause(0.25)


