import soundfile
from scipy.spatial.distance import correlation as correlate
from scipy.fft import fft
from scipy import stats
import numpy as np
import statistics
from collections import OrderedDict
import librosa



# Allocate samples to arrays of their respective recordings. Note that all bitrates are assumed to be the same because the samples were all recorded 
# during the same session with the same recording equipment.
# FIXME take these recordings
a, bitrate     = (soundfile.read('a_0.wav'), soundfile.read('a_1.wav'), soundfile.read('a_2.wav'), soundfile.read('a_3.wav'), soundfile.read('a_4.wav'))
b, bitrate     = (soundfile.read('b_0.wav'), soundfile.read('b_1.wav'), soundfile.read('b_2.wav'), soundfile.read('b_3.wav'), soundfile.read('b_4.wav'))
c, bitrate     = (soundfile.read('c_0.wav'), soundfile.read('c_1.wav'), soundfile.read('c_2.wav'), soundfile.read('c_3.wav'), soundfile.read('c_4.wav'))
d, bitrate     = (soundfile.read('d_0.wav'), soundfile.read('d_1.wav'), soundfile.read('d_2.wav'), soundfile.read('d_3.wav'), soundfile.read('d_4.wav'))
e, bitrate     = (soundfile.read('e_0.wav'), soundfile.read('e_1.wav'), soundfile.read('e_2.wav'), soundfile.read('e_3.wav'), soundfile.read('e_4.wav'))
f, bitrate     = (soundfile.read('f_0.wav'), soundfile.read('f_1.wav'), soundfile.read('f_2.wav'), soundfile.read('f_3.wav'), soundfile.read('f_4.wav'))
g, bitrate     = (soundfile.read('g_0.wav'), soundfile.read('g_1.wav'), soundfile.read('g_2.wav'), soundfile.read('g_3.wav'), soundfile.read('g_4.wav'))
h, bitrate     = (soundfile.read('h_0.wav'), soundfile.read('h_1.wav'), soundfile.read('h_2.wav'), soundfile.read('h_3.wav'), soundfile.read('h_4.wav'))
i, bitrate     = (soundfile.read('i_0.wav'), soundfile.read('i_1.wav'), soundfile.read('i_2.wav'), soundfile.read('i_3.wav'), soundfile.read('i_4.wav'))
j, bitrate     = (soundfile.read('j_0.wav'), soundfile.read('j_1.wav'), soundfile.read('j_2.wav'), soundfile.read('j_3.wav'), soundfile.read('j_4.wav'))
k, bitrate     = (soundfile.read('k_0.wav'), soundfile.read('k_1.wav'), soundfile.read('k_2.wav'), soundfile.read('k_3.wav'), soundfile.read('k_4.wav'))
l, bitrate     = (soundfile.read('l_0.wav'), soundfile.read('l_1.wav'), soundfile.read('l_2.wav'), soundfile.read('l_3.wav'), soundfile.read('l_4.wav'))
m, bitrate     = (soundfile.read('m_0.wav'), soundfile.read('m_1.wav'), soundfile.read('m_2.wav'), soundfile.read('m_3.wav'), soundfile.read('m_4.wav'))
n, bitrate     = (soundfile.read('n_0.wav'), soundfile.read('n_1.wav'), soundfile.read('n_2.wav'), soundfile.read('n_3.wav'), soundfile.read('n_4.wav'))
o, bitrate     = (soundfile.read('o_0.wav'), soundfile.read('o_1.wav'), soundfile.read('o_2.wav'), soundfile.read('o_3.wav'), soundfile.read('o_4.wav'))
p, bitrate     = (soundfile.read('p_0.wav'), soundfile.read('p_1.wav'), soundfile.read('p_2.wav'), soundfile.read('p_3.wav'), soundfile.read('p_4.wav'))
q, bitrate     = (soundfile.read('q_0.wav'), soundfile.read('q_1.wav'), soundfile.read('q_2.wav'), soundfile.read('q_3.wav'), soundfile.read('q_4.wav'))
r, bitrate     = (soundfile.read('r_0.wav'), soundfile.read('r_1.wav'), soundfile.read('r_2.wav'), soundfile.read('r_3.wav'), soundfile.read('r_4.wav'))
s, bitrate     = (soundfile.read('s_0.wav'), soundfile.read('s_1.wav'), soundfile.read('s_2.wav'), soundfile.read('s_3.wav'), soundfile.read('s_4.wav'))
t, bitrate     = (soundfile.read('t_0.wav'), soundfile.read('t_1.wav'), soundfile.read('t_2.wav'), soundfile.read('t_3.wav'), soundfile.read('t_4.wav'))
u, bitrate     = (soundfile.read('u_0.wav'), soundfile.read('u_1.wav'), soundfile.read('u_2.wav'), soundfile.read('u_3.wav'), soundfile.read('u_4.wav'))
v, bitrate     = (soundfile.read('v_0.wav'), soundfile.read('v_1.wav'), soundfile.read('v_2.wav'), soundfile.read('v_3.wav'), soundfile.read('v_4.wav'))
w, bitrate     = (soundfile.read('w_0.wav'), soundfile.read('w_1.wav'), soundfile.read('w_2.wav'), soundfile.read('w_3.wav'), soundfile.read('w_4.wav'))
x, bitrate     = (soundfile.read('x_0.wav'), soundfile.read('x_1.wav'), soundfile.read('x_2.wav'), soundfile.read('x_3.wav'), soundfile.read('x_4.wav'))
y, bitrate     = (soundfile.read('y_0.wav'), soundfile.read('y_1.wav'), soundfile.read('y_2.wav'), soundfile.read('y_3.wav'), soundfile.read('y_4.wav'))
z, bitrate     = (soundfile.read('z_0.wav'), soundfile.read('z_1.wav'), soundfile.read('z_2.wav'), soundfile.read('z_3.wav'), soundfile.read('z_4.wav'))
space, bitrate = (soundfile.read('__0.wav'), soundfile.read('__1.wav'), soundfile.read('__2.wav'), soundfile.read('__3.wav'), soundfile.read('__4.wav'))

alphabet = (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, space)
alphabet_strings = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ')



# Values identified in algorithm
n_1 = 5     # Samples per letter
n_2 = 3     # Number of most correlated keys to include
n_3 = 3     # Error correction depth 
n_4 = 3     # Error correction iterations



# Returns the n_2 most correlated keys of a sample
def correlate(sample) -> OrderedDict:
    correlations = dict()
    threshold_z = 2

    # Check sample against each possible key
    for letter in range(len(alphabet)):

        # Check spectrum of sample against spectrum of each recording of the letter
        for recording in range(len(alphabet[letter])):
            correlations[alphabet_strings[letter]] = correlate(fft(sample), fft(alphabet[letter][recording]))

        # Outlier detection before mean taken
        outlier_indices = np.where(np.abs(stats.zscore(correlations[alphabet_strings[letter]])) > threshold_z)[0]
        correlations = dict(filter(lambda item: item[0] not in outlier_indices, correlations.items()))

        correlations[alphabet_strings[letter]] = statistics.mean(correlations[alphabet[letter]])

    most_correlated = OrderedDict(sorted(correlations.items(), key=lambda item: item[1]))[:n_2]
    return list(most_correlated.items())



# Sample of secret message. Bitrate assumed to be equal to those of letters due to recording with the same equipment.
secret_message, bitrate = soundfile.read('secret_message.wav')



# Split the message up into individual keystrokes via transient detection
onset_frames = librosa.onset.onset_detection(y=secret_message, sr=bitrate, backtrack=True)
keystrokes = ()
for idx in range(len(onset_frames)):
    if idx < len(onset_frames) - 1:
        keystrokes.append(secret_message[onset_frames[idx]:onset_frames[idx + 1]])
    else:
        keystrokes.append(secret_message[onset_frames[idx]:])



# Get correlated keys. Form is list of duple ('letter_string', correlation), sorted by correlation amt
correlated_keys = ()
for keystroke in keystrokes:
    correlated_keys.append(correlate(keystroke))



# Display message 
# FIXME no error detection implemented
decrypted_message = ()
for key in correlated_keys:
    decrypted_message.append(key[0])
print(decrypted_message)