# Signal-Processing
Code for signal processing related subjects, in BME dept. ITS
## Signal Generation
Two code provided in this repository consist of the basic stationary signal and non-stationary signal generation. Stationary signal means that the signal is periodic, which repeats in specific time period (T). In the code, I implement two types of wave to choose (sine and cosine), using Numpy library. Modify the amplitude, frequency (Hz), and overall duration in the code to understand the characteristic of the signal. You can also change the rate, which is the amount of data computed in 1 second. For non-stationary signal, the signal is not periodic (obviously did not repeats). Same as stationary signal, I also implement two types of wave to choose, but able to modify how many different signal in overall duration, the amplitude and frequency of each signal, and also the rate.
## DFT
Discrete Fourier Transform (DFT) algorithm works by doing fourier transform (FT) on the discrete signal data, able to see the frequency component. In the code, I implement two version of the DFT, the long version and short version. The purpose of the long version is to easily understand what is going on with the DFT equation, the looping iteration, the multiplication of the exponential number, etc. The short version is the faster approach of the code, which the looping iteration implemented by matrix multiplication, which is more efficient than the long version. In the code, DFT used to know the frequency component of stationary signal, which is able to easily verivy the result.
## STFT
To be continue..
## CWT
To be continue..
## Downsampling
To be continue..
