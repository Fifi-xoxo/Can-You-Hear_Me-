✨CAN YOU HEAR ME?✨

A python based audio steganography system that hides a secret audio message within a cover audio file using frequency 
domain processing. This project was developed as part of my MSc research into audio steganography and how data can 
be concealed using psychoacoustic methods whilst keeping similarities between the original and stego output.

--------------------------------------------------------------------------------------------------------------------------
Capabilities

* Conceal a secret message within a cover audio file
* Use STFT to work in frequency domain
* Embed data into highest frequency peaks
* Extract hidden data
* Reconstruct the audio using inverse STFT
* Calculate the SNR to evaluate the outputted stego audio
* Visualise cover vs stego waveforms

--------------------------------------------------------------------------------------------------------------------------
Architecture

* Audio Preparation: Loads the cover and secret .wav files and forces them to the exact same array length using
  trimming or zero-padding so they match up perfectly.
  
* Frequency Conversion: Converts the left and right channels from the time-domain into the frequency-domain using
   STFT so the code can look at the spectrogram shapes.
  
* Loudest Peak Embedding: Finds the 5 loudest frequencies in each time frame and injects the secret data into them
   using a 0.1 scaling factor to keep the hidden message completely inaudible to the human ear.
  
* Stereo Reconstruction: Turns the modified frequencies back into time-domain signals using ISTFT and
  stacks the left and right channels back together into a final stereo .wav file.
  
* Secret Extraction: Compares the original cover audio with the stego audio, finds the exact peak coordinates
  used during embedding, and applies a reverse calculation to extract and rebuild the hidden secret.
  
* SNR Quality Check: Runs the audio through a Signal-to-Noise Ratio (SNR) formula to calculate the exact difference
   between the cover and the stego file to measure the success of the program.
--------------------------------------------------------------------------------------------------------------------------
Motivation

I developed this project as part of my MSc research into audio steganography and its capabilities. I was interested in exploring
how information could be concealed within an audio signal without significantly altering the characteristics of the original 
audio. This project was created to combine my passion for cyber security, signal processing, and creative technology.

--------------------------------------------------------------------------------------------------------------------------
Future Developments 

For future developments I would firstly integrate a web based interface hat supports drag-and-drop functionality 
for both the cover audio and the secret message. This would take away the inconvenience of having to have these files sit directly
within the project folder. Users could then easily interchange, experiment with, and compare different audio assets in real time.
