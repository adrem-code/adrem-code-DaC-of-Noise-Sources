import numpy as np
import librosa
import scipy.fftpack as fft
from scipy.signal import get_window, windows


def RMSnormalization(signal, desired_RMS_lvl):
    """
    RMSnormalization function normalizes signal to desired RMS level.

    :signal: numpy audio signal vector
    :desired_RMS_lvl: desired RMS value in dB
    :return: signal normalized to desired RMS level
    """
    #Convert to linear scale
    R = 10**(desired_RMS_lvl/20)

    scaling_factor = np.sqrt((len(signal)*R**2)/sum(signal**2))
    output = signal*scaling_factor

    return output


def frame_audio(audio, frame_len, hop_size, sample_rate):
    """
    frame_audio function divides signal to frames with set fft length and overlap.

    :audio: numpy mono audio signal vector
    :frame_len: length of frame
    :hop_size: overlap; described in no of samples
    :return: 2D numpy array returning another frame data
    """ 
    audio = np.pad(audio, int(frame_len / 2), mode='reflect')
    frame_num = int((len(audio) - frame_len) / hop_size) + 1
    
    frames = np.zeros((frame_num,frame_len))
    for n in range(frame_num):
        frames[n] = audio[n*hop_size:n*hop_size+frame_len]
    
    return frames


def find_boundary_fc(f_min, f_max, ERB_scale_factor = 1):
    """
    find_boundary_fc function calculates center frequencies of boundary filters.

    :f_min: lowest frequency of filterbank
    :f_max: highest frequency of filterbank
    """ 
    a = 6.23e-6*ERB_scale_factor
    b= 93.39e-3*ERB_scale_factor
    c= 28.52*ERB_scale_factor

    am = 0.5/(700+f_min)
    bm = 700/(700+f_min)
    cm = -0.5*f_min*(1+700/(700+f_min))

    bv = (b-bm)/(a-am)
    cv = (c-cm)/(a-am)

    fc_min = 0.5*(-bv+np.sqrt(bv**2 -4*cv))

    am = -0.5/(700+f_max)
    bm = -700/(700+f_max)
    cm = f_max*(1+700/(700+f_max))/2

    bv = (b-bm)/(a-am)
    cv = (c-cm)/(a-am)

    fc_max = 0.5*(-bv+np.sqrt(bv**2 -4*cv))

    return fc_min, fc_max


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def get_HFCC_filter_points(fmin, fmax, filter_num):
    """
    get_HFCC_filter_points function calculates center frequencies of remaining filters

    :fmin: center frequency of lowest filter in Hz
    :fmax: center frequency of highest filter in Hz
    :mel_filter_num: number of filters
    :return: 1D array containing center frequencies of all filters in Hz
    """ 
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)

    filter_fcs = [fmin_mel]
    for i in range(2,filter_num):
        filter_fcs.append(fmin_mel +(i-1)*\
            ((fmax_mel-fmin_mel)/(filter_num-1)))
    filter_fcs.append(fmax_mel)
    filter_fcs = np.array(filter_fcs)

    freqs = mel_to_freq(filter_fcs)
    
    return freqs


def freq_limits_of_filters(freqs,  ERB_scale_factor=1):
    """
    freq_limits_of_filters function calculates upper and lower frequencies of filters

    :freqs: numpy 1D array with filterbank center frequencies in Hz
    :return: 1D array containing lower frequencies of all filters in Hz
             1D array containing upper frequencies of all filters in Hz
    """ 
    centerFmel = freq_to_mel(freqs)
    ERB = 6.23*(freqs/1000)**2 + 93.39*(freqs/1000) + 28.52
    ERB = ERB*ERB_scale_factor

    filter_fls = 2595*np.log10(-2/1400*ERB+0.5*\
        np.sqrt((2/700*ERB)**2+4*10**(2*centerFmel/2595)))
    filter_fus = 2*centerFmel-filter_fls
    filter_fls = mel_to_freq(filter_fls)
    filter_fus = mel_to_freq(filter_fus)

    return filter_fls, filter_fus


def get_HFCC_filters(freq_min, freq_high, FFT_size, sample_rate,mel_filter_num=128,ERB_scale_factor=1):
    """
    get_HFCC_filters function calculates HFCC filterbank

    :freq_min: lowest frequency of filterbank
    :freq_high: highest frequency of filterbank
    :FFT_size: length of FFT sequence
    :mel_filter_num: no of ERB-spaced mel filters
    :return: numpy array with following filters
    """ 
    f_min, f_max = find_boundary_fc(freq_min, freq_high, ERB_scale_factor)
    filter_fcs = get_HFCC_filter_points(f_min, f_max, mel_filter_num)
    filter_fls, filter_fus = freq_limits_of_filters(filter_fcs, ERB_scale_factor)

    filter_fcs = np.floor((FFT_size+1)*(filter_fcs/sample_rate)).astype(int)
    filter_fls = np.floor((FFT_size+1)*(filter_fls/sample_rate)).astype(int)
    filter_fus = np.floor((FFT_size+1)*(filter_fus/sample_rate)).astype(int)
    
    filters = np.zeros((len(filter_fcs),int(FFT_size/2+1)))

    for n in range(len(filter_fcs)):
        filters[n, filter_fls[n] : filter_fcs[n]] = \
            np.linspace(0, 1, filter_fcs[n]-filter_fls[n])
        filters[n, filter_fcs[n] : filter_fus[n]] = \
            np.linspace(1, 0, filter_fus[n]-filter_fcs[n])
    
    return filters


def get_MFCC_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate):
    """
    get_MFCC_filter_points function calculates center frequencies of MFCC filters

    :fmin: lowest frequency of filterbank
    :fmax: highest frequency of filterbank
    :mel_filter_num: number of filters
    :FFT_size:length of FFT sequence
    :return: 1D array containing center frequencies of all filters in Hz
    """ 
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    hz = mel_to_freq(mels)

    freqs = np.floor((FFT_size + 1) / sample_rate * hz).astype(int)
    
    return freqs

def get_MFCC_filters(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate):
    """
    get_MFCC_filters function calculates HFCC filterbank

    :freq_min: lowest frequency of filterbank
    :freq_high: highest frequency of filterbank
    :FFT_size: length of FFT sequence
    :mel_filter_num: no of ERB-spaced mel filters
    :return: numpy array with following filters
    """ 
    filter_points = get_MFCC_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate)

    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis


def singletaper_windowing(audio_framed, FFT_size, window_type = 'hann'):

    window = get_window(window_type, FFT_size, fftbins=True)
    audio_win = audio_framed * window
    audio_winT = np.transpose(audio_win)

    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

    audio_fft = np.transpose(audio_fft)
    audio_power = np.square(np.abs(audio_fft))

    return audio_power


def multitaper_windowing(audio_framed, FFT_size, NW = 2.5, M=5):
    """
    multitaper_windowing function uses Slepian sequence to window signal's frames and calculates following FFT's

    :audio_framed: 2D array containing framed signal
    :FFT_size: len of FFT
    :NW: standarized half bandwidth
    :M: no of slepian's windows
    :return: 2D array power spectrum [no of frame, FFT_size]
    """ 
    tapers = windows.dpss(FFT_size, 2.5, 5, return_ratios=True)[0]
    audio_multi = np.transpose(audio_framed)
    
    audio_multitaper = np.empty((int(1+ FFT_size // 2), audio_multi.shape[1]), order='F')
    for n in range(audio_multi.shape[1]):
        audio_tapers = np.empty((int(1+ FFT_size // 2), tapers.shape[0]), order='F')
        for m in range(tapers.shape[0]): 
            win_sig = audio_multi[:, n]*tapers[m,:]
            audio_fft = fft.fft(win_sig)
            audio_fft = audio_fft[:int(FFT_size//2+1)]
            audio_power = np.square(np.abs(audio_fft))
            audio_tapers[:,m] = audio_power
        audio_taper = np.mean(audio_tapers, axis = 1)
        audio_multitaper[:, n] = audio_taper
    audio_multitaper = np.transpose((audio_multitaper))
    
    return audio_multitaper


def hfcc(audio, sample_rate, win_len, freq_min, freq_high,method='single',FFT_size = 2048, hop_size = 512, dct_filter_num = 14,mel_filter_num = 128,  ERB_scale_factor = 1):
    """
    hfcc function calculates Human Factor Cepstral Coefficients from audio signal

    :audio: array of signal samples
    :freq_min: bottom frequency of filterbank
    :freq_high: upper frequency of filterbank
    :method: 'single' to calculate spectrogram with hann window; 'multi' to calculate spectrogram with Slepian sequence
    :hop_size: overlap in signal segmentation
    :dct_filter_num: no of required cepstral coefficient vectors
    :mel_filter_num: no of required ERB-spaced mel filters
    :return: 2D array of cepstral coefficients, 2D array spectrum filtered with ERB-spaced mel filterbank
    """ 

    #Dividing audio on frames with overlap   
    audio_framed = frame_audio(audio, frame_len=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
    
    if method == 'single':
        #calculating PSD with hann window
        audio_power = singletaper_windowing(audio_framed, FFT_size=FFT_size)
        
    if method == 'multi':
        #calculating PSD with multitaper method
        audio_multitaper = multitaper_windowing(audio_framed=audio_framed, FFT_size=FFT_size) 

    #calculating HFCC filters
    ERB_filters = get_HFCC_filters(freq_min, freq_high,FFT_size=FFT_size, sample_rate = sample_rate,mel_filter_num=mel_filter_num, ERB_scale_factor=ERB_scale_factor)

    #Normalizing Filters
    filters = librosa.util.normalize(ERB_filters, norm=1, axis=-1)

    #Filtering spectrum
    if method == 'single':
        audio_filtered = np.dot(filters, np.transpose(audio_power))
    if method == 'multi':
        audio_filtered = np.dot(filters, np.transpose(audio_multitaper))
    audio_log = librosa.power_to_db(audio_filtered, ref = np.max)

    #Calculating DCT to obtain HFCC
    dct_filters = dct(dct_filter_num, mel_filter_num)
    cepstral_coefficents = np.dot(dct_filters, audio_log)

    return cepstral_coefficents, audio_log


def mfcc(audio, sample_rate, win_len, freq_min, freq_high,method = 'single',FFT_size = 2048, hop_size = 512, dct_filter_num = 14,mel_filter_num = 128):
    """
    mfcc function calculates Mel Frequency Cepstral Coefficients from audio signal

    :audio: array of signal samples
    :freq_min: bottom frequency of filterbank
    :freq_high: upper frequency of filterbank
    :method: 'single' to calculate spectrogram with hann window; 'multi' to calculate spectrogram with Slepian sequence
    :hop_size: overlap in signal segmentation
    :dct_filter_num: no of required cepstral coefficient vectors
    :mel_filter_num: no of required ERB-spaced mel filters
    :return: 2D array of cepstral coefficients, 2D array spectrum filtered with ERB-spaced mel filterbank
    """ 
    
    #Dividing audio on frames with overlap   
    audio_framed = frame_audio(audio, frame_len=FFT_size, hop_size=hop_size, sample_rate=sample_rate)

    if method == 'single':
        #calculating PSD with hann window
        audio_power = singletaper_windowing(audio_framed, FFT_size=FFT_size)
        
    if method == 'multi':
        #calculating PSD with multitaper method
        audio_multitaper = multitaper_windowing(audio_framed=audio_framed, FFT_size=FFT_size)

    filters = get_MFCC_filters(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=sample_rate)

    filters = librosa.util.normalize(filters, norm=1, axis=-1)

    #Logarithm
    if method == 'single':
        audio_filtered = np.dot(filters, np.transpose(audio_power))
    if method == 'multi':
        audio_filtered = np.dot(filters, np.transpose(audio_multitaper))
    audio_log = librosa.power_to_db(audio_filtered, ref = np.max)

    #Calculating DCT
    dct_filters = dct(dct_filter_num, mel_filter_num)
    cepstral_coefficents = np.dot(dct_filters, audio_log)

    return cepstral_coefficents, audio_log