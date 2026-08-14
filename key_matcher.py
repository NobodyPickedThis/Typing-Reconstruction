import soundfile
from scipy.spatial.distance import correlation as correlate
from scipy.signal.windows import tukey
from scipy.fft import fft
from scipy import stats
import numpy as np
import statistics
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr

# =================CONSTANTS=AND=SAMPLES=================

# Values identified in algorithm
n_1 = 5     # Samples per letter
n_2 = 3     # Number of most correlated keys to include
n_3 = 3     # Error correction depth 
n_4 = 3     # Error correction iterations

DO_ERROR_CORRECTION = True

def to_mono(signal):
    return signal[:, 0] if signal.ndim > 1 else signal

# For denoising
noise, bitrate = soundfile.read('noise.wav')
noise = to_mono(noise)
denoise_thresh = 0.5 # std deviations away from noise
denoise_amt = 0.95   # 1 -> max attenuation, 0 -> no attenuation

# Allocate samples to arrays of their respective recordings. Note that all bitrates are assumed to be the same because the samples were all recorded 
# during the same session with the same recording equipment.
a_data     = [soundfile.read(f'a_{i}.wav') for i in range(1, n_1 + 1)]
b_data     = [soundfile.read(f'b_{i}.wav') for i in range(1, n_1 + 1)]
c_data     = [soundfile.read(f'c_{i}.wav') for i in range(1, n_1 + 1)]
d_data     = [soundfile.read(f'd_{i}.wav') for i in range(1, n_1 + 1)]
e_data     = [soundfile.read(f'e_{i}.wav') for i in range(1, n_1 + 1)]
f_data     = [soundfile.read(f'f_{i}.wav') for i in range(1, n_1 + 1)]
g_data     = [soundfile.read(f'g_{i}.wav') for i in range(1, n_1 + 1)]
h_data     = [soundfile.read(f'h_{i}.wav') for i in range(1, n_1 + 1)]
i_data     = [soundfile.read(f'i_{i}.wav') for i in range(1, n_1 + 1)]
j_data     = [soundfile.read(f'j_{i}.wav') for i in range(1, n_1 + 1)]
k_data     = [soundfile.read(f'k_{i}.wav') for i in range(1, n_1 + 1)]
l_data     = [soundfile.read(f'l_{i}.wav') for i in range(1, n_1 + 1)]
m_data     = [soundfile.read(f'm_{i}.wav') for i in range(1, n_1 + 1)]
n_data     = [soundfile.read(f'n_{i}.wav') for i in range(1, n_1 + 1)]
o_data     = [soundfile.read(f'o_{i}.wav') for i in range(1, n_1 + 1)]
p_data     = [soundfile.read(f'p_{i}.wav') for i in range(1, n_1 + 1)]
q_data     = [soundfile.read(f'q_{i}.wav') for i in range(1, n_1 + 1)]
r_data     = [soundfile.read(f'r_{i}.wav') for i in range(1, n_1 + 1)]
s_data     = [soundfile.read(f's_{i}.wav') for i in range(1, n_1 + 1)]
t_data     = [soundfile.read(f't_{i}.wav') for i in range(1, n_1 + 1)]
u_data     = [soundfile.read(f'u_{i}.wav') for i in range(1, n_1 + 1)]
v_data     = [soundfile.read(f'v_{i}.wav') for i in range(1, n_1 + 1)]
w_data     = [soundfile.read(f'w_{i}.wav') for i in range(1, n_1 + 1)]
x_data     = [soundfile.read(f'x_{i}.wav') for i in range(1, n_1 + 1)]
y_data     = [soundfile.read(f'y_{i}.wav') for i in range(1, n_1 + 1)]
z_data     = [soundfile.read(f'z_{i}.wav') for i in range(1, n_1 + 1)]
space_data = [soundfile.read(f'__{i}.wav') for i in range(1, n_1 + 1)]

a     = [to_mono(d) for d, sr in a_data]
b     = [to_mono(d) for d, sr in b_data]
c     = [to_mono(d) for d, sr in c_data]
d     = [to_mono(d) for d, sr in d_data]
e     = [to_mono(d) for d, sr in e_data]
f     = [to_mono(d) for d, sr in f_data]
g     = [to_mono(d) for d, sr in g_data]
h     = [to_mono(d) for d, sr in h_data]
i     = [to_mono(d) for d, sr in i_data]
j     = [to_mono(d) for d, sr in j_data]
k     = [to_mono(d) for d, sr in k_data]
l     = [to_mono(d) for d, sr in l_data]
m     = [to_mono(d) for d, sr in m_data]
n     = [to_mono(d) for d, sr in n_data]
o     = [to_mono(d) for d, sr in o_data]
p     = [to_mono(d) for d, sr in p_data]
q     = [to_mono(d) for d, sr in q_data]
r     = [to_mono(d) for d, sr in r_data]
s     = [to_mono(d) for d, sr in s_data]
t     = [to_mono(d) for d, sr in t_data]
u     = [to_mono(d) for d, sr in u_data]
v     = [to_mono(d) for d, sr in v_data]
w     = [to_mono(d) for d, sr in w_data]
x     = [to_mono(d) for d, sr in x_data]
y     = [to_mono(d) for d, sr in y_data]
z     = [to_mono(d) for d, sr in z_data]
space = [to_mono(d) for d, sr in space_data]

bitrate = a_data[0][1]
alphabet = (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, space)
alphabet_strings = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ')

for letter in alphabet:
    for pos in range(len(letter)):
        letter[pos] = nr.reduce_noise(y=letter[pos], sr=bitrate, y_noise=noise, stationary=True, n_std_thresh_stationary=denoise_thresh, prop_decrease=denoise_amt)

# =================CORRELATION=FUNCTIONS=================

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


# =================MAIN=LOGIC=================

# Sample of secret message. Bitrate assumed to be equal to those of letters due to recording with the same equipment.
secret_message, bitrate = soundfile.read('secret_message_1.wav')
secret_message = to_mono(secret_message)

# Denoise
secret_message = nr.reduce_noise(y=secret_message, sr=bitrate, y_noise=noise, stationary=True, n_std_thresh_stationary=denoise_thresh, prop_decrease=denoise_amt)


# Split the message up into individual keystrokes via transient detection
onset_frames = librosa.onset.onset_detect(
    y=secret_message, 
    sr=bitrate, 
    backtrack=True, 
    units='samples',
    hop_length=256,
    delta=0.1,
    wait=int(0.15 * bitrate / 256),
    pre_max=int(0.03 * bitrate / 256),
    post_max=int(0.03 * bitrate / 256)
    )

# =================VISUALIZATION=================

plt.figure(figsize=(14, 4))
plt.plot(np.linspace(start=1, stop=len(secret_message), num=len(secret_message)), secret_message, lw=0.5, color='red')
plt.vlines(onset_frames, ymin=-1, ymax=1)
plt.ylim(bottom=-1,top=1)
plt.title("Onset Detections")
plt.show()

# ===============================================

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


# Initial message attempt
decrypted_keys = list()
for key in correlated_keys:
    decrypted_keys.append(key[0][0])
decrypted_message = ''.join(decrypted_keys)

if not DO_ERROR_CORRECTION:
    print(decrypted_message)
    exit()

# =================ERROR=CORRECTION=================

# Store any valid detected words to display as suggestions
# in case error correction on the larger message fails.
all_valid_words = set()

# Detect if message contains all valid words
def is_valid(message) -> bool:
    text = ''.join(message)
    words_in_message = text.split(sep=' ')
    with open('words.txt') as words_file:
        valid = True
        words_list = set(words_file.read().split())
        for word in words_in_message:
            if not word in words_list:
                valid = False
            else:
                all_valid_words.add(word)
        return valid


# Makes a single change, swapping the letter with the least confidence that has a 
# relatively high confidence second guess. Letters may be excluded from this process,
# done during higher correction depths.
def tweak_message(correlations, excluded_indices):
    # Create a list of non-excluded indices
    candidate_indices = [i for i in range(len(correlations)) if i not in excluded_indices]

    # Return early if no further tries to be made (word length < iter depth)
    if not candidate_indices:
        return correlations, excluded_indices

    # Generate 3 candidates with small gaps between their first and second guesses
    best_candidates = list()
    for z in range(3):
        best_candidates.append(max(candidate_indices, key=lambda i: correlations[i][0][1] - correlations[i][1][1]))
        candidate_indices.remove(best_candidates)
    # Winning candidate is one with lowest confidence first guess
    final_candidate = max(candidate_indices, key=lambda i: correlations[i][0][1])

    # Drop the chosen candidate's first guess
    correlations[final_candidate].pop(0)

    # Add chosen candidate to excluded values and return new correlations
    excluded_indices.append(final_candidate)
    return correlations, excluded_indices


current_depth = 0
current_iteration = 0

# Display message if valid
if is_valid(decrypted_message):
    print(decrypted_message)

else:
    current_iteration += 1
    current_depth += 1

    # Initial change of single letter
    altered_correlations, excluded_indices = tweak_message(correlated_keys, [])

    # Check if the correction was successful
    altered_message = list()
    for key in altered_correlations:
        altered_message.append(key[0][0])

    # This loop resets the correlations to their original values and
    # excludes the value previously tried 
    while not is_valid(altered_message) and current_iteration <= n_4:
        current_iteration += 1

        # Call the tweak function with the original list of keys
        #  
        # Previous attempts' exclusions will be remembered by excluded 
        # values, without also excluding all exclusions made in the
        # inner loop
        altered_correlations, excluded_indices = tweak_message(correlated_keys, excluded_indices)
        altered_message = list()
        for key in altered_correlations:
            altered_message.append(key[0][0])

        # This loop iterates the same process on the version of the
        # correlations with only a single change
        while not is_valid(altered_message) and current_depth <= n_3:
            excluded_this_iter = []
            current_depth += 1
            altered_correlations, excluded_this_iter = tweak_message(altered_correlations, excluded_indices + excluded_this_iter)
            altered_message = list()
            for key in altered_correlations:
                altered_message.append(key[0][0])

    # Display message, indicate if invalid
    if is_valid(altered_message):
        print(altered_message)
    else:
        print("Unable to validate message.\n\tFirst attempt:", decrypted_message, "\n\tLast attempt:", altered_message)
        print("All valid words generated during correction attempt:", all_valid_words)
            