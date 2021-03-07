import json
import os
import math
import librosa
import numpy as np
import CepstralCoefficients


DATASET_PATH = 'D:\\Praca Inzynierska\\Nagrania posegregowane\\'#main folder with recordings
JSON_PATH = "C:\\AED\\MEL_single.json"#where to save coefficients
SAMPLE_RATE = 22050
win_len = 0.1 #in sec

def save_mfcc(dataset_path, json_path, n_ceps=14, n_fft=2048, hop_length=512, num_segments=3799, method_of_windowing = 'single', type_of_parameters = 'mfcc', n_filter = 40):
    '''
    dataset_path - main folder with recordings
    json_path - where to save coeffitiens, end with name_of_file.json
    method of frames - 'single' for hann, 'multi' for DPSS
    type of parameters - 'mel', 'erb', 'mfcc', 'hfcc'
    n_filter - number of mel/erb filters
    n_ceps - number of mfcc/hfcc vectors
    '''
    
    # dictionary to store mapping, labels, and parameters
    data = {
        "mapping": [],
        "labels": [],
        "parameters": []
    }

    iter = 0
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            
            print("Liczba probek: " + str(iter))
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            
            iter = 0
            # process all audio files in genre sub-dir
            for f in filenames:
                if iter >num_segments:
                    break
		    # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(int(len(signal)/win_len/SAMPLE_RATE)):
                    
                    # calculate start and finish sample for current segment
                    start = int(win_len*SAMPLE_RATE * d)
                    finish = int(start + win_len*SAMPLE_RATE)

                    if finish <= len(signal):
                        # extract mfcc
                        if iter >num_segments:
                            break
                        #normalize frame to equal RMS value
                        signal_norm = CepstralCoefficients.RMSnormalization(signal[start:finish], -1)
                        #for mel spectrogram
                        if type_of_parameters == 'mel':
                            _, parameters = CepstralCoefficients.mfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
                        #for erb spectrogram
                        if type_of_parameters == 'erb':
                            _, parameters = CepstralCoefficients.hfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
                        #for mfcc
                        if type_of_parameters == 'mfcc':
                            parameters,_ = CepstralCoefficients.mfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
                        #for hfcc
                        if type_of_parameters == 'hfcc':
                            parameters,_ = CepstralCoefficients.hfcc(signal_norm, sample_rate=sample_rate,win_len=n_fft,FFT_size = n_fft, hop_size = hop_length,freq_min=0, freq_high=11025, method=method_of_windowing, mel_filter_num=n_filter, dct_filter_num=n_ceps)
                        print("Features Shape: " + str(parameters.shape))
                        iter +=1
                        
                        # store only mfcc feature with expected number of vectors
                        
                        data["parameters"].append(parameters.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    print(iter)
    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)