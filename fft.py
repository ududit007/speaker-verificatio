import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
# rate, data = wav.read('aa.wav')

def calc_fft(data):
	fft_out = fft(data)
	return fft_out
