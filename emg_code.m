filePath = "C:\Users\shwet\Desktop\DSP PROJECT\Electro-Myography-EMG-Dataset\raw_emg_data_unprocessed\index_finger_motion_raw.csv";

if isfile(filePath)
    emg_data = readmatrix(filePath);
    emg_signal = emg_data';
else
    error('File not found');
end

Fs = 1000; % Sampling frequency
Fc_high = 20; % High-pass filter cutoff frequency
Fc_low = 499; % Low-pass filter cutoff frequency

[b_high, a_high] = butter(2, Fc_high/(Fs/2), 'high');
filtered_emg = filtfilt(b_high, a_high, emg_signal);

[b_low, a_low] = butter(2, Fc_low/(Fs/2), 'low');
filtered_emg = filtfilt(b_low, a_low, filtered_emg);

features = zeros(size(emg_signal, 2), 8); 
for i = 1:size(emg_signal, 2)
  features(i, 1) = calculate_max_fractal_length(emg_signal(:, i));
  features(i, 2) = calculate_mm_average(emg_signal(:, i));
  features(i, 3) = calculate_dasdv(emg_signal(:, i));
  features(i, 4) = calculate_avg_amplitude_change(emg_signal(:, i));
  features(i, 5) = calculate_enhanced_wavelength(emg_signal(:, i));
  features(i, 6) = calculate_fft_variance(emg_signal(:, i));
  features(i, 7) = calculate_fft_max_intensity(emg_signal(:, i));
  features(i, 8) = calculate_neo_variance(emg_signal(:, i));
end

writematrix(features,'Extreacted_features.csv');

% Feature extraction functions

% Maximum Fractal Length (MFL)
function mfl = calculate_max_fractal_length(signal)
  N = length(signal);
  diff_signal = diff(signal);
  mfl = log10(sqrt(sum(diff_signal.^2) / (N-1)));
end

function mmavg = calculate_mm_average(signal)
  window_size = 5; % Adjust window size as needed
  if length(signal) > 12
    mmavg = mean(abs(filtfilt(ones(1,window_size)/window_size, 1, signal)));
  else
    mmavg = mean(abs(signal)); 
  end
end

function dasdv = calculate_dasdv(signal)
  mean_val = mean(signal);
  diff_from_mean = signal - mean_val;
  dasdv = sqrt(var(diff_from_mean));
end

% Average Amplitude Change (AAC)
function aac = calculate_avg_amplitude_change(signal)
  aac = mean(abs(diff(signal)));
end

% Enhanced Wavelength
function ewl = calculate_enhanced_wavelength(signal)
  carrier_freq = 100; % Replace with appropriate carrier frequency
  carrier_signal = sin(2*pi*carrier_freq*(1:length(signal))/length(signal));
  modulated_signal = signal .* carrier_signal;
  ewl = sum(abs(diff(modulated_signal))); % Wavelength after modulation
  ewl = mean(ewl); % Ensure a single value is returned
end


% FFT Variance
function fft_var = calculate_fft_variance(signal)
  fft_values = fft(signal);
  fft_power_spectrum = abs(fft_values).^2;
  fft_var = var(fft_power_spectrum);
end

% FFT Maximum Intensity
function fft_max_intensity = calculate_fft_max_intensity(signal)
  fft_values = fft(signal);
  fft_power_spectrum = abs(fft_values).^2;
  fft_max_intensity = max(fft_power_spectrum);
end

% Variance of NEO
function neo_var = calculate_neo_variance(signal)
  neo_output = zeros(size(signal));
  for i = 2:length(signal)-1
    neo_output(i) = signal(i)^2 - signal(i-1)*signal(i+1);
  end
  neo_var = var(neo_output);
end
