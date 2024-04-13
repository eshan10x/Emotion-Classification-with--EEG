import os
from flask import Flask, render_template, request, jsonify, flash, make_response
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask_cors import CORS
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.signal import resample
import numpy as np
import scipy.io as sio
import os
import warnings
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from pywt import wavedec
import scipy.signal
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from glob import glob
from tqdm import tqdm, trange
import re
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)

# Directory where uploaded files will be saved
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'npy'}
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

channels_to_use = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return make_response('No file part', 400)  # Return a 400 Bad Request for error scenarios
    
    file = request.files['file']
    if file.filename == '':
        return make_response('No selected file', 400)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        data = loadmat(get_latest_file('/Users/eshan/Documents/Uni/Year 4/FYP/EEG Implementation/Flask/uploads'))
        # Extract the data you need, for example, 'X' variable
        keys = get_keynames(data.keys())
        X = data[keys[1]].squeeze()  # Assuming X is a 1D array, use squeeze to remove singleton dimensions

        sampling_rate = 200
        window_size = 2
        total_samples = sampling_rate * window_size
        channel_id_to_plot = 0
        channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

        X = np.random.randn(1, 14, total_samples)

        # Extracting raw time series data from the dataset
        raw_ts = X[0, channel_names.index(channels_to_use[channel_id_to_plot]), 0:total_samples]

        df = pd.DataFrame({
            'Time': np.linspace(0, window_size, num=total_samples),  # Create a time axis in seconds
            'Voltage': raw_ts  # EEG data values
        })

        # Create the plot with Plotly Express
        fig = px.line(df, x='Time', y='Voltage', title='Comparison of an Exemplary Time Series',
                    labels={'Time': 'Time t [s]', 'Voltage': 'Voltage U [mV]'},
                    markers=True)

        graph_json = fig.to_json()
        return graph_json

#Get latest file from uploaded files
def get_latest_file(directory):
    #"Get latest file from saved location"
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    # Filter out directories, leaving only files
    files = [f for f in files if os.path.isfile(f)]
    # Sort the files by modification time
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Return the first item in the list, which is the latest file
    if files:
        latest_file = files[0]
        return latest_file
    else:
        return None

#Get key names when selecting one signal section
def get_keynames(dict_keys):
        dict_keys = list(dict_keys)
        trial_names = list()
        for trial_name in dict_keys:
            for i in range(1,16):
                if '_eeg'+str(i) in trial_name:
                    trial_names.append(trial_name)
        return trial_names

@app.route('/classify', methods=['POST'])

def classification():

    ######### LSTM Model #########
    def lstm_model(input_shape, clip_value=1.0):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    ######### CNN Model #########
    def build_cnn_model(input_shape):
        model = Sequential([
            Input(shape=input_shape),

            # First Convolutional Block
            Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.5),

            # Second Convolutional Block
            Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.5),

            # Third Convolutional Block (optional)
            Conv1D(filters=256, kernel_size=2, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.5),

            # Flattening and Final Dense Layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')  # Assuming 3 classes
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


    ######### Combined Model #########
    def build_combined_model(cnn_model, lstm_model, input_shape_cnn, input_shape_lstm, n_classes):
        # Define input layers for CNN and LSTM
        cnn_input = Input(shape=input_shape_cnn)
        lstm_input = Input(shape=input_shape_lstm)

        # Use the CNN and LSTM models to process the inputs
        cnn_features = cnn_model(cnn_input)
        lstm_features = lstm_model(lstm_input)

        # Combine the features from both models
        combined_features = concatenate([cnn_features, lstm_features])

        # Add final layers for classification
        x = Dense(64, activation='relu')(combined_features)
        x = Dropout(0.5)(x)
        final_output = Dense(n_classes, activation='softmax')(x)

        # Create the combined model
        combined_model = Model(inputs=[cnn_input, lstm_input], outputs=final_output)

        # Compile the combined model
        combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return combined_model

    ####### Feature Extraction #######
    ##Calculates the energy of wavelet coefficients, either for an array of coefficients or single value.
    def compute_energy(coefficients):
        if isinstance(coefficients, np.ndarray):
            return np.sum(np.square(np.abs(coefficients))) / len(coefficients)
        elif np.isscalar(coefficients):
            return np.square(np.abs(coefficients))
        else:
            raise ValueError("Unsupported type for coefficients")

    def compute_total_energy(approximation_coefficients, detail_coefficients):
        total_energy = 0
        total_energy += compute_energy(approximation_coefficients)
        for detail_coefficient in detail_coefficients:
            total_energy += compute_energy(detail_coefficient)
        return total_energy

    def calculate_D_Energy(detail_coefficients):
        total_energy = 0
        for detail_coefficient in detail_coefficients:
            total_energy += compute_energy(detail_coefficient)
        return total_energy


    def compute_mean(coefficients):
        return np.mean(coefficients)

    def compute_std(coefficients):
        return np.std(coefficients)

    def calculate_D_mean(coeffs):
        valid_indices = [i for i in range(1, min(6, len(coeffs)))]
        return np.mean([np.mean(coeffs[i]) for i in valid_indices])


    def calculate_A_mean(coeffs):
        return compute_mean(coeffs[0])

    def calculate_D_std(coeffs):
        return np.mean([compute_std(coeffs[i]) for i in range(min(6, len(coeffs)))])

    def calculate_A_std(coeffs):
        return compute_std(coeffs[0])


    def wavelet_feature_extraction(data, type_wav, sampling_frequency, nperseg=256):
        coefficients = wavedec(data, type_wav, level=5)

        total_energy = compute_total_energy(coefficients[0], coefficients[1:])
        cD_Energy=calculate_D_Energy(coefficients[1:])
        cA_Energy=compute_energy(coefficients[0])
        cD_mean = calculate_D_mean(coefficients[1:])
        cA_mean = calculate_A_mean(coefficients[0])
        cD_std = calculate_D_std(coefficients[1:])
        cA_std = calculate_A_std(coefficients[0])

        return [
            total_energy,
            cD_Energy,
            cA_Energy,
            cD_mean,
            cA_mean,
            cD_std,
            cA_std,

        ]

    def get_median_frequency(psd):
        median_frequency = np.median(psd)

        return median_frequency

    def get_edge_frequency(psd):
        edge_frequency = np.where(psd >= psd.max() / 2)[0][0]

        return edge_frequency

    def compute_power_spectral_density(data, sampling_frequency, nperseg=256):
        _, psd = scipy.signal.welch(data, fs=sampling_frequency, nperseg=nperseg)
        return psd

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = scipy.signal.lfilter(b, a, data)
        return y

    def compute_band_power(psd_result, freq_band_indices, fs, nperseg):
        freq_band_power = np.sum(psd_result[freq_band_indices]) * fs / nperseg
        return freq_band_power

    def compute_spectral_entropy(psd):
        normalized_psd = psd / np.sum(psd)  # Normalize to obtain probabilities
        spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd))
        return spectral_entropy

    def extract_frequency_domain_features(signal, sampling_frequency, nperseg=256):
        # Apply Butterworth bandpass filters
        delta_band_signal = butter_bandpass_filter(signal, 0.5, 4, sampling_frequency)
        theta_band_signal = butter_bandpass_filter(signal, 4, 8, sampling_frequency)
        alpha_band_signal = butter_bandpass_filter(signal, 8, 13, sampling_frequency)
        beta_band_signal = butter_bandpass_filter(signal, 13, 30, sampling_frequency)
        gamma_band_signal = butter_bandpass_filter(signal, 30, 40, sampling_frequency)

        # Compute Power Spectral Density for each band
        delta_psd = compute_power_spectral_density(delta_band_signal, sampling_frequency, nperseg=nperseg)
        theta_psd = compute_power_spectral_density(theta_band_signal, sampling_frequency, nperseg=nperseg)
        alpha_psd = compute_power_spectral_density(alpha_band_signal, sampling_frequency, nperseg=nperseg)
        beta_psd = compute_power_spectral_density(beta_band_signal, sampling_frequency, nperseg=nperseg)
        gamma_psd = compute_power_spectral_density(gamma_band_signal, sampling_frequency, nperseg=nperseg)

        # Compute Band Power for each frequency band
        freq_band_indices = [range(int(nperseg * band[0] / sampling_frequency), int(nperseg * band[1] / sampling_frequency)) for band in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]]

        delta_band_power = compute_band_power(delta_psd, freq_band_indices[0], sampling_frequency, nperseg)
        theta_band_power = compute_band_power(theta_psd, freq_band_indices[1], sampling_frequency, nperseg)
        alpha_band_power = compute_band_power(alpha_psd, freq_band_indices[2], sampling_frequency, nperseg)
        beta_band_power = compute_band_power(beta_psd, freq_band_indices[3], sampling_frequency, nperseg)
        gamma_band_power = compute_band_power(gamma_psd, freq_band_indices[4], sampling_frequency, nperseg)

        spectral_entropy_result = compute_spectral_entropy(np.concatenate([delta_psd, theta_psd, alpha_psd, beta_psd, gamma_psd]))
        # Compute the power spectral density (PSD)
        psd, _ = scipy.signal.welch(signal, fs=sampling_frequency, nperseg=nperseg)

        return [
            delta_band_power,
            theta_band_power,
            alpha_band_power,
            beta_band_power,
            gamma_band_power,
            spectral_entropy_result,
        ]

    def compute_standard_deviation(data):
        return np.std(data)

    def compute_skewness(data):
        return skew(data)

    def compute_kurtosis(data):
        return kurtosis(data)

    def compute_median(data):
        return np.median(data)

    def compute_band_power_time(data, sampling_frequency, nperseg):
        _, power_density = welch(data, fs=sampling_frequency, nperseg=nperseg)
        return np.mean(power_density)
    def peak_to_peak_voltage(data):
        return np.ptp(data)

    def total_signal_area(data):
        return np.sum(np.abs(data))

    def decorrelation_time(data):
        autocorrelation = np.correlate(data, data, mode='full')
        zero_crossings = np.where(np.diff(np.sign(autocorrelation)))[0]

        if len(zero_crossings) > 0:
            first_zero_crossing = zero_crossings[0]
            time_points = np.arange(len(autocorrelation))
            decorrelation_time = time_points[first_zero_crossing]
            return decorrelation_time
        else:
            return -1
    def extract_time_domain_features(raw_data,sampling_frequency, nperseg=256):
        # data=butter_bandpass_filter(raw_data, 0.5, 40, sampling_frequency)
        data=raw_data

        features = [
            compute_standard_deviation(data),
            compute_skewness(data),
            compute_kurtosis(data),
            compute_median(data),
            compute_band_power_time(data, sampling_frequency, nperseg),
            peak_to_peak_voltage(data),
            total_signal_area(data),
            decorrelation_time(data)
        ]
        return features


    def create_feature_dataset(data_amps):
        data_tensor = []
        channel_data = []
        for channel in tqdm(data_amps, desc="Extracting featuers..."):

            wavelet_featuers = wavelet_feature_extraction(channel, "db4", 128)
            frequency_featuers = extract_frequency_domain_features(channel,128)
            time_featuers = extract_time_domain_features(channel,128)
            all_featuers = wavelet_featuers+frequency_featuers+time_featuers
            channel_data.append(all_featuers)

        data_tensor.append(channel_data)
        return np.array(data_tensor)


    ######### Pre Processing EEG Signal #########
    baseline_removal_window = 3
    cutoff_frequencies = [4,40]
    seconds_to_use = 185
    downsampling_rate = 128
    #channels_to_use = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    window_size = 2
    window_overlap = 0
    save_plots_to_file = False
    sampling_rate = 200
    channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

    #Applying Filters
    def butter_bandpass(lowcut, highcut, fs, btype='band', order=5):
            nyq = 0.5 * fs
            if btype == 'bandpass':
                low = lowcut / nyq
                high = highcut / nyq
                sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
            elif btype == 'highpass':
                low = lowcut / nyq
                sos = butter(order, low, analog=False, btype='highpass', output='sos')
            elif btype == 'lowpass':
                high = highcut / nyq
                sos = butter(order, high, analog=False, btype='lowpass', output='sos')
            return sos

    def butter_bandpass_filter(X, lowcut, highcut, fs, btype='bandpass', order=5):
            sos = butter_bandpass(lowcut, highcut, fs, btype=btype, order=order)
            X = sosfilt(sos, X)
            return X

    #Down Sampling
    def down_sample(data, downsampling_rate, sampling_rate):
        if not(downsampling_rate == 0) and not(downsampling_rate == sampling_rate):
            new_length = int(data.shape[1] / sampling_rate * downsampling_rate)
            data_downsampled = np.zeros((data.shape[0], new_length))

            for channel_id in range(data.shape[0]):
                data_downsampled[channel_id, :] = resample(data[channel_id, :], new_length)
            return data_downsampled

    #Selet certain channels
    def select_channel(data, channels_to_use,channel_names):
        if channels_to_use == None:
            channels_to_use = channel_names
        channel_index_list = list()
        for i in range(len(channels_to_use)):
            if channels_to_use[i] in channel_names:
                channel_index_list.append(channel_names.index(channels_to_use[i]))
            else:
                warnings.warn(' Channel ' + channels_to_use[i] +' could not be found in the list of actual channels')

        data_selected_channels = np.zeros((len(channels_to_use), data.shape[1]))
        for channel in trange(len(channel_index_list)):
            data_selected_channels[channel,:] = data[channel_index_list[channel],:]
        return data_selected_channels

    #Cut into windows
    def cut_into_windows(data):
        window_size = 2
        window_overlap = 0

        num_points_per_window = window_size * downsampling_rate
        num_points_overlap = window_overlap * downsampling_rate
        stride = num_points_per_window - num_points_overlap
        start_index = [0]
        end_index = [num_points_per_window]
        num_windows_per_exp = 1
        while(end_index[-1]+stride < data.shape[1]):
            num_windows_per_exp = num_windows_per_exp + 1
            start_index.append(start_index[-1] + stride)
            end_index.append(end_index[-1] + stride)
        data_cut = np.zeros((data.shape[0], num_points_per_window))

        for window_id in range(len(start_index)):
            data_cut[:,:] = data[:,start_index[window_id]:end_index[window_id]]

        data = data_cut
        return data


    def load_mat(file_path, session_name):
        segment_data = sio.loadmat(file_path)
        keys = get_keynames(segment_data.keys())
        data = segment_data[session_name]


        if not(baseline_removal_window==0):
            print(f"[+] baseline removal starting.. ")
            baseline_datapoints = baseline_removal_window * sampling_rate

            baseline = data[:,:baseline_datapoints].sum(1) / baseline_datapoints

            for timestep in trange(data.shape[1]):
                data[:,timestep] = data[:,timestep] - baseline

        if not(cutoff_frequencies[0] == None):
            if not(cutoff_frequencies[1] == None):
                btype='bandpass'
            else:
                btype='highpass'
        elif not (cutoff_frequencies[1] == None):
                btype='lowpass'


        print(f"[+] Applying filters to data..")

        for channel_id in range(data.shape[0]):
            data[channel_id, :] = butter_bandpass_filter(
                                                            data[channel_id, :],
                                                            cutoff_frequencies[0],
                                                            cutoff_frequencies[1],
                                                            sampling_rate,
                                                            btype=btype,
                                                            order=5)


        if not(seconds_to_use == None):
            num_sample_points_to_use = seconds_to_use * sampling_rate
            data_selected = data[:,:num_sample_points_to_use]
            print(f"[+] data selection widnows {num_sample_points_to_use}")

        print(f"[+] data downsampling..")
        down_sampled_data = down_sample(data_selected,downsampling_rate, sampling_rate)

        print(f"[+] channel selection..")
        select_channel_data  = select_channel(down_sampled_data,channels_to_use,channel_names)

        print(f"[+] cuting into windows..")
        cuts = cut_into_windows(select_channel_data)
        print(f"[+] data pre-process completed..")

        return cuts

    #file_path = "/content/drive/MyDrive/FYP/Dataset/SEED/SEED_EEG/Preprocessed_EEG/9_20140627.mat"
    file_path = get_latest_file('/Users/eshan/Documents/Uni/Year 4/FYP/EEG Implementation/Flask/uploads')
    segment_data = sio.loadmat(file_path)
    keys = get_keynames(segment_data.keys())
    
    dataset = load_mat(file_path, keys[1])

    print(f"dataset : {dataset.shape}")

    featuers = create_feature_dataset(dataset)
    print(f"[+] featuer extracted data: {featuers.shape}")

    scaler = StandardScaler()
    input_features = scaler.fit_transform(featuers.reshape(-1, featuers.shape[-1])).reshape(featuers.shape)

    input_shape = (14,21)
    lstm_model = lstm_model(input_shape)
    cnn_model = build_cnn_model(input_shape)


    # Load your pre-trained models here
    lstm_model.load_weights('/Users/eshan/Documents/Uni/Year 4/FYP/EEG Implementation/Flask/lstm_model_weights.h5')
    cnn_model.load_weights('/Users/eshan/Documents/Uni/Year 4/FYP/EEG Implementation/Flask/cnn_model_weights.h5')

    cnn_pred = cnn_model.predict(input_features)
    lstm_pred = lstm_model.predict(input_features)
    combined_pred = (cnn_pred + lstm_pred) / 2

    predicted_class = np.argmax(combined_pred, axis=1)

    emotion_labels = ['Happy', 'Sad', 'Angry']

    print(predicted_class)

    predicted_emotion = emotion_labels[predicted_class[0]]

    print(f"Predicted Emotion: {predicted_emotion}")

    # return predicted_emotion
    return make_response(f"{predicted_emotion}", 200)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about-us')
def about_us():
    return render_template('about_us.html')


if __name__ == '__main__':
    app.run(debug=True)
