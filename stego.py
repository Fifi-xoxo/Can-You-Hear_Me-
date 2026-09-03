import librosa 
import matplotlib.pyplot as plot 
import numpy as np 
import soundfile as sf 

cover_audioarray, cover_samplerate = sf.read("cover.wav") 
secret_audioarray, secret_samplerate = sf.read("secret_audio.wav") 

if secret_audioarray.shape[0] > cover_audioarray.shape[0]: 
    secret_audioarray = secret_audioarray[:cover_audioarray.shape[0], :] 
else:
    pad_length = cover_audioarray.shape[0] - secret_audioarray.shape[0] 
    secret_audioarray = np.pad(secret_audioarray, ((0, pad_length), (0, 0)), mode='constant') 

cover_audio = cover_audioarray.astype(np.float32) 
secret_audio = secret_audioarray.astype(np.float32) 
#the first code block prepares the audio data by converting both files into identical array lengths
# and data types. If the secret audio is too long then the excess is trimmed, but if the secret
#audio is too short, zero padding is added to the end until the length matches the cover audio.

n_fft = 1024 
hop_length = 256 
scale = 0.1 

S_cover_L = librosa.stft(cover_audio[:,0], n_fft=n_fft, hop_length=hop_length) 
S_cover_R = librosa.stft(cover_audio[:,1], n_fft=n_fft, hop_length=hop_length) 
S_secret_L = librosa.stft(secret_audio[:,0], n_fft=n_fft, hop_length=hop_length) 
S_secret_R = librosa.stft(secret_audio[:,1], n_fft=n_fft, hop_length=hop_length) 
#parameters are set before the embedding process takes place.1024/4 = 256, is used for better reconstruction
#later on. The audio arrays are currently in time-domain but it would be beneficial to assess the
#frequency-domain, allowing the secret to be hidden in the loudset frequencies.
# The hopes is viewing the freuency-domain will allow for the secret to be imperceptible 
#to the human ear. 

def embed_secret(STFT_cover, STFT_secret, scale=0.1): 
    STFT_stego = STFT_cover.copy() 
    magnitude = np.abs(STFT_cover) 
    high_freq_bins = 5 
    n_freq, n_time = magnitude.shape
    for t in range(n_time): 
        strongest = np.argpartition(magnitude[:, t], -high_freq_bins)[-high_freq_bins:] 
        for idx in strongest: 
            if idx < n_freq and t < STFT_secret.shape[1]: 
                STFT_stego[idx, t] += STFT_secret[idx, t] * scale                   
    return STFT_stego 
#The embedding function hides the scret audios spectogram within the cover audios spectogram.
#To preserve the integrity of the data, the original file is copied rather than overwritten,
#and its magnitude spectrum is calculated. For each time frame, rhe 5 highest frequencies are
#chosen and embeded with the secret message because these dominant frequencies are more likely
#to provide optimal psychoacoustic masking, making the secret message inaudible to the human ear.

def extract_secret(STFT_cover, STFT_stego, scale=0.1, high_freq_bins=5): 
    n_freq, n_time = STFT_cover.shape 
    STFT_secret_recovered = np.zeros_like(STFT_cover, dtype=np.complex64) 
    magnitude = np.abs(STFT_cover) 
    for t in range(n_time): 
        strongest = np.argpartition(magnitude[:, t], -high_freq_bins)[-high_freq_bins:] 

        for idx in strongest: 
            if idx < n_freq and t < STFT_stego.shape[1]:
                    STFT_secret_recovered[idx, t] = (STFT_stego[idx, t] - STFT_cover[idx, t]) / scale 
    return STFT_secret_recovered 
#The extraction function retrieves the hidden secret spectrogram by comparing the original
#cover audio with the modified stego-audio. AN empty array is then created to store the recovered
#data based on the covers dimensions. The algorithm calculates the magnitude of the original
#cover to locate the exact points in which the secret message has been hidden during the embedding
#function. The secret can then be extracted and reconstructed.

S_stego_L = embed_secret(S_cover_L, S_secret_L, scale)  
S_stego_R = embed_secret(S_cover_R, S_secret_R, scale) 
stego_L = librosa.istft(S_stego_L, hop_length=hop_length) 
stego_R = librosa.istft(S_stego_R, hop_length=hop_length) 
stego_audio = np.vstack((stego_L, stego_R)).T 
stego_audio /= np.max(np.abs(stego_audio)) 

sf.write("stego_out.wav", stego_audio, cover_samplerate) 
print("Stego audio saved as stego_out.wav") 
#The embedding function is executed independently across both the left and right channels
#to maintain a complete stero field. After the secret is embedded, the inverse shot-time 
#fourier transform is pplied to reconstruct both signal back into time domain. Both channels
#are stacked and recombined into a standard stereo array. 

def calculate_snr(cover, stego): 
    min_len = min(len(cover), len(stego)) 
    cover = cover[:min_len]
    stego = stego[:min_len]

    if cover.ndim > 1:
        cover = np.mean(cover, axis=1) 
        stego = np.mean(stego, axis=1) 
        
    noise = cover - stego 
    signal_power = np.sum(cover ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr 

#The above function calculates the signal to noise ratio to distinguish the level of distortion
#introduced by the embedding process, the formula used was outlined by Aslantaş & Hanilçi (2022)
#To make sure that it remains compatible, the lengths of the cover and stego signals are trimmed to match
#If multi-channel audio is discovered, both channels are merged. The calculation is then preformed,
#to distinguish the noise difference between the cover and stego audio. Finally, the sum of squares
#is used to find the total power of both the signal and the noise, then the result is coverted into decibels.

cover_audioarray, _ = sf.read("cover.wav")
stego_audioarray, _ = sf.read("stego_out.wav")

snr_value = calculate_snr(cover_audioarray, stego_audioarray) 
print(f"SNR Value: {snr_value:.2f} dB") 

print("Starting extraction...")

cover_audioarray, _ = sf.read("cover.wav") 
stego_audioarray, _ = sf.read("stego_out.wav")

cover_audio = cover_audioarray.astype(np.float32)
stego_audio = stego_audioarray.astype(np.float32)

S_cover_L = librosa.stft(cover_audio[:,0], n_fft=n_fft, hop_length=hop_length)
S_cover_R = librosa.stft(cover_audio[:,1], n_fft=n_fft, hop_length=hop_length)

S_stego_L = librosa.stft(stego_audio[:,0], n_fft=n_fft, hop_length=hop_length)
S_stego_R = librosa.stft(stego_audio[:,1], n_fft=n_fft, hop_length=hop_length)

S_secret_rec_L = extract_secret(S_cover_L, S_stego_L, scale)
S_secret_rec_R = extract_secret(S_cover_R, S_stego_R, scale)

secret_rec_L = librosa.istft(S_secret_rec_L, hop_length=hop_length)
secret_rec_R = librosa.istft(S_secret_rec_R, hop_length=hop_length)

secret_rec_audio = np.vstack((secret_rec_L, secret_rec_R)).T 
secret_rec_audio /= np.max(np.abs(secret_rec_audio))
#The above code block executes the final verification phase of this program. Firstly, the SNR
#is calculated and displayed to measure the transparacy of the hidden message within the cover audio.
#The extraction process then begins by taking both the cover and generated stego audio files
#into the frequency domain via STFT. The extraction algorithm works separtatly acroos both left
#and right channels, calculating the differential frequency content. Once recovered, ISTFT rebuilds
#the hidden audio back into standdard stero array.  

sf.write("recovered_secret.wav", secret_rec_audio, cover_samplerate)
print("Recovered secret saved as recovered_secret.wav")

plot.figure(figsize=(12,4))
plot.plot(cover_audioarray[:,0], label="Original Cover Audio - Left")
plot.plot(stego_audio[:,0], alpha=0.6, label="Stego Audio - Left")
plot.title("Original Cover vs Stego Audio Waveform")
plot.legend()
plot.show()
#the last code block lets the user know the secret has been recovered successfully and saved. A graph is then plotted
#using the left channel of the original audio vs the stego audio. This is done so that a user can compare the 
#simulaties between audio outputs (helps to visual assess the success of the program)

