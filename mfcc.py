import numpy as np
from scipy.io import wavfile

# Get file
wavfile_path = 'file.wav'
sample_rate, signal = wavfile.read(wavfile_path)

# In MS
frame_length = 0.025
frame_skip = 0.01

# In Samples
frame_length_samples = int(0.025*sample_rate)
frame_skip_samples = int(0.01*sample_rate)
n_samples = len(signal)

# Find number of frames
n_frames = 1 + int((n_samples-frame_length_samples)//frame_skip_samples)

# Pad signal to be length of n_frames
n_padded_samples = ((n_frames*frame_skip_samples) + frame_length_samples) - n_samples
padding = np.zeros((n_padded_samples))
padded_signal = np.append(signal,padding)

# Set frames
frames = np.zeros((n_frames, frame_length_samples))
starting_index = 0
for frame_no in range(n_frames):
    frames[frame_no] = padded_signal[starting_index:starting_index+frame_length_samples]
    starting_index += frame_skip_samples

# Define hamming window
hamming = np.zeros((frame_length_samples))
for n in range(frame_length_samples):
    hamming[n] = 0.54 - 0.46*np.cos((2*np.pi*n)/(frame_length_samples-1))

# Apply hamming window to signal
filtered_frames = np.zeros((n_frames, frame_length_samples))
for frame_no in range(n_frames):
    filtered_frames[frame_no] = frames[frame_no]*hamming

# STFT of signal (FFT of each frame, only keep first half(real))
n_fft = frame_length_samples
n_fft_real = int(n_fft//2)+1
stft = np.zeros((n_frames,n_fft))
for frame_no in range(n_frames):
    for k in range(n_fft_real):
        for n in range(n_fft):
            stft[frame_no,k] += (filtered_frames[frame_no,n] * np.exp((-j*2*np.pi*k*n)/n_fft))



# Compute power spectrum (generally use 512 fft and keep first 257 coefficients?)
power_spec = ((np.absolute(n_fft))**2)/n_fft

def freq_to_mel(frequency):
    return 1125*np.ln(1+(frequency/700))

def mel_to_freq(mel):
    return 700*(np.exp(mel/1125)-1)

# Create mel filters (26 is standard)
# m = 2595 log_{10}(1+(f/700))
# f = 700(10^{m/2595}-1)

min_frequency = 300
max_frequency = int(sample_rate//2)
n_banks = 26
n_mels = n_frequencies = n_bins = n_banks+2
min_mel = freq_to_mel(min_frequency)
max_mel = freq_to_mel(max_frequency)

# Create mels list (+2 for endpoints)
mels = np.linspace(min_mel,max_mel,n_mels)
frequencies = np.zeros((n_frequencies))
bins = np.zeros((n_bins))

# Create frequencies list
for i in range(n_frequencies):
    frequencies[i] = mel_to_freq(mels[i])

# This is equivalent to (((n_fft_real+1)*frequency)//max_frequency), which makes
# more sense
def freq_to_bin(frequency,n_fft,sample_rate):
    return (((n_fft+1)*frequency)//sample_rate)

for i in range(n_bins):
    bins[i] = freq_to_bin(frequencies[i],n_fft,sample_rate)

n_filters = 40
filter_bank = np.zero
