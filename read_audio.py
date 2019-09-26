from scipy.io.wavfile import read
import os

def read_audio(file):
	sr,signal=read(file)
	try:
		signal = signal[:,0]
	except:
		pass
	# print(signal[:10])
	# print('time',len(signal)/sr)
	signal=signal/abs(max(signal))
	# print('sampling rate',sr)
	return sr,signal