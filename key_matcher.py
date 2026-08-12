import soundfile
from scipy.spatial.distance import correlation as correlate
from scipy.signal.windows import tukey
from scipy.fft import fft
from scipy import stats
import numpy as np
import statistics
import librosa


# Values identified in algorithm
n_1 = 5     # Samples per letter
n_2 = 3     # Number of most correlated keys to include
n_3 = 3     # Error correction depth 
n_4 = 3     # Error correction iterations


# Allocate samples to arrays of their respective recordings. Note that all bitrates are assumed to be the same because the samples were all recorded 
# during the same session with the same recording equipment. Samples expected to be mono.
# FIXME take these recordings
a_data     = [soundfile.read(f'a_{i}.wav') for i in range(n_1)]
b_data     = [soundfile.read(f'b_{i}.wav') for i in range(n_1)]
c_data     = [soundfile.read(f'c_{i}.wav') for i in range(n_1)]
d_data     = [soundfile.read(f'd_{i}.wav') for i in range(n_1)]
e_data     = [soundfile.read(f'e_{i}.wav') for i in range(n_1)]
f_data     = [soundfile.read(f'f_{i}.wav') for i in range(n_1)]
g_data     = [soundfile.read(f'g_{i}.wav') for i in range(n_1)]
h_data     = [soundfile.read(f'h_{i}.wav') for i in range(n_1)]
i_data     = [soundfile.read(f'i_{i}.wav') for i in range(n_1)]
j_data     = [soundfile.read(f'j_{i}.wav') for i in range(n_1)]
k_data     = [soundfile.read(f'k_{i}.wav') for i in range(n_1)]
l_data     = [soundfile.read(f'l_{i}.wav') for i in range(n_1)]
m_data     = [soundfile.read(f'm_{i}.wav') for i in range(n_1)]
n_data     = [soundfile.read(f'n_{i}.wav') for i in range(n_1)]
o_data     = [soundfile.read(f'o_{i}.wav') for i in range(n_1)]
p_data     = [soundfile.read(f'p_{i}.wav') for i in range(n_1)]
q_data     = [soundfile.read(f'q_{i}.wav') for i in range(n_1)]
r_data     = [soundfile.read(f'r_{i}.wav') for i in range(n_1)]
s_data     = [soundfile.read(f's_{i}.wav') for i in range(n_1)]
t_data     = [soundfile.read(f't_{i}.wav') for i in range(n_1)]
u_data     = [soundfile.read(f'u_{i}.wav') for i in range(n_1)]
v_data     = [soundfile.read(f'v_{i}.wav') for i in range(n_1)]
w_data     = [soundfile.read(f'w_{i}.wav') for i in range(n_1)]
x_data     = [soundfile.read(f'x_{i}.wav') for i in range(n_1)]
y_data     = [soundfile.read(f'y_{i}.wav') for i in range(n_1)]
z_data     = [soundfile.read(f'z_{i}.wav') for i in range(n_1)]
space_data = [soundfile.read(f'__{i}.wav') for i in range(n_1)]

a     = [d for d, sr in a_data]
b     = [d for d, sr in b_data]
c     = [d for d, sr in c_data]
d     = [d for d, sr in d_data]
e     = [d for d, sr in e_data]
f     = [d for d, sr in f_data]
g     = [d for d, sr in g_data]
h     = [d for d, sr in h_data]
i     = [d for d, sr in i_data]
j     = [d for d, sr in j_data]
k     = [d for d, sr in k_data]
l     = [d for d, sr in l_data]
m     = [d for d, sr in m_data]
n     = [d for d, sr in n_data]
o     = [d for d, sr in o_data]
p     = [d for d, sr in p_data]
q     = [d for d, sr in q_data]
r     = [d for d, sr in r_data]
s     = [d for d, sr in s_data]
t     = [d for d, sr in t_data]
u     = [d for d, sr in u_data]
v     = [d for d, sr in v_data]
w     = [d for d, sr in w_data]
x     = [d for d, sr in x_data]
y     = [d for d, sr in y_data]
z     = [d for d, sr in z_data]
space = [d for d, sr in space_data]

bitrate = a_data[0][1]  # assuming all recordings share the same samplerate, per your comment
alphabet = (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, space)
alphabet_strings = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ')


# Used for preparing samples for spectral comparison
# Alpha controls how strong the taper is: 1 -> rect, 0 -> half-Hann
def window_and_pad(signal, target_len, alpha=0.5):
    windowed = signal * tukey(len(signal) * 2, alpha)[len(signal):]
    if len(windowed) < target_len:
        windowed = np.pad(windowed, (0, target_len - len(windowed)))
    return windowed


# Returns the n_2 most correlated keys of a sample
def correlate_key(sample):
    correlations = dict()
    threshold_z = 2

    # Check sample against each possible key
    for letter in range(len(alphabet)):

        # Prepare sample(s) for accurate comparison
        recordings = alphabet[letter]
        target_len = max(len(sample), max(len(rec) for rec in recordings))
        padded_sample = window_and_pad(sample, target_len)
        padded_recordings = [window_and_pad(rec, target_len) for rec in recordings]

        # Check spectrum of sample against spectrum of each recording of the letter
        correlation_scores = [correlate(np.abs(fft(padded_sample)), np.abs(fft(padded_recordings[r]))) for r in range(len(padded_recordings))]

        # Outlier detection before mean taken
        correlation_scores = np.array(correlation_scores)
        z = np.abs(stats.zscore(correlation_scores))
        clean_scores = correlation_scores[z <= threshold_z]
        if len(clean_scores) == 0:
            clean_scores = correlation_scores
        correlations[alphabet_strings[letter]] = statistics.mean(clean_scores)

    most_correlated = sorted(correlations.items(), key=lambda item: item[1])[:n_2]
    return most_correlated


# Sample of secret message. Bitrate assumed to be equal to those of letters due to recording with the same equipment.
secret_message, bitrate = soundfile.read('secret_message.wav')


# Split the message up into individual keystrokes via transient detection
onset_frames = librosa.onset.onset_detect(y=secret_message, sr=bitrate, backtrack=True, units='samples')
keystrokes = list()
for idx in range(len(onset_frames)):
    if idx < len(onset_frames) - 1:
        keystrokes.append(secret_message[onset_frames[idx]:onset_frames[idx + 1]])
    else:
        keystrokes.append(secret_message[onset_frames[idx]:])


# Get correlated keys. Form is list of duple ('letter_string', correlation), sorted by correlation amt
correlated_keys = list()
for keystroke in keystrokes:
    correlated_keys.append(correlate_key(keystroke))


# Display message 
# FIXME no error detection implemented
decrypted_message = list()
for key in correlated_keys:
    decrypted_message.append(key[0][0])
print(decrypted_message)