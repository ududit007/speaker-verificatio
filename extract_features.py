import pyaudio
import wave
import sys
import time
import librosa
import numpy as np 
import numpy
import matplotlib.pyplot as plt 
import parselmouth as pm
from scipy.spatial.distance import euclidean
from parselmouth.praat import call
from read_audio import *
from short_term_energy import *
from fft import *
from scipy.io.wavfile import write as write
from fastdtw import fastdtw as dtw


def padding(x,y):
	if len(x)>len(y):
		maxLen=len(x)
	else:
		maxLen=len(y)
	y=np.append(y,np.zeros(maxLen-len(y)))
	x=np.append(x,np.zeros(maxLen-len(x)))
	return x,y


def find_mfcc(voiceId, numOfCoeff=12, windowLength=0.025, timeStep=0.005, firstFilterMel = 100.0, DistancebetweenFilters=100.0,
		maxFreq=0.0):
	mfcc = call(voiceId,"To MFCC", numOfCoeff, windowLength, timeStep, firstFilterMel, DistancebetweenFilters, maxFreq)
	return mfcc.to_array()

def find_features(signal,sampling_rate,frame_size,frame_stride,window_type='hamming'):
	"""
	signal [array/list] - the audio files in the form of an array which contains amplitude values against time
	samping_rate [float]- the number of samples per second (44Khz for example means 44000 samples/second)
	frame_len [integer] - the length of the window 
	frame_size [integer] - the overlap between two windows
	window_type [string] - the type of window that will be applied, default value is a hamming window
	
	return value : a json file with arrays of feautres
	"""
	#convert a list to numpy array for ease of manipulation and calculation
	# signal=np.array(signal)

	# Convert from seconds to samples
	frame_length, frame_step = frame_size * sampling_rate, frame_stride * sampling_rate
	signal_length = len(signal)
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	
	# Make sure that we have at least 1 frame
	num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) 

	#to ensure that there are no partially filled frames
	pad_signal_length = num_frames * frame_step + frame_length
	z=np.zeros(pad_signal_length-signal_length)
	
	#the padded signal after adding necessary zeros
	padded_signal=np.append(signal,z)
	indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
	frames = padded_signal[indices.astype(np.int32, copy=False)]
	# print('number  of frames', len(frames))

	#check the type of windowing to be done
	if window_type=='hamming':
		frames*=np.hamming(frame_length)
		# frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))



	############################################################################################################
	# Exctracting features 
	############################################################################################################
		
	#short term energy
	ste=short_term_energy(frames)
	ffts=calc_fft(frames)
	# print(ste)
	#average short term energy
	# avg_ste=[sum(frame)/len(frame) for frame in ste]
	
	# plt.figure(figsize=(10,5))
	# plt.subplot(3,1,1)
	# plt.title('Amplitude')
	# plt.plot(signal)
	# plt.subplot(3,1,2)
	# plt.title('Short term energy')
	# plt.plot(ste)

	# plt.subplot(3,1,3)
	# plt.title('FFT')
	# plt.plot(ffts)

	# plt.show()
	return signal, ste, ffts

def record_audio(filename):
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 3
	WAVE_OUTPUT_FILENAME = filename

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
		
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK)

	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	



def main(frame_size=0.0231999,frame_stride=0.010):
	#take command line argument as the file name
	filename1=input('Give a voice print for enrollment:')
	print('reading audio file')
	time.sleep(1)
	record_audio(filename1)
	sampling_rate,signal=read_audio(filename1)
	sound = pm.Sound(filename1)
	# print(type(signal))
	# print(signal)
	print('Feature extraction started')
	time.sleep(1)
	print('Calculating amplitude...')
	time.sleep(1)
	amp1, e1, fft1=find_features(signal,sampling_rate,frame_size,frame_stride,'hamming')
	print('calculating energy...')
	time.sleep(1)
	print('calculating FFT...')
	time.sleep(1)
	print("MFCC______________________________")
	time.sleep(1)
	print('Feature extraction completed')
	time.sleep(3)

	#print("MFCC______________________________")
	mfcc1=find_mfcc(sound)
	#print(find_mfcc(sound))
	# plt.figure(figsize=(10,5))
	# plt.subplot(3,1,1)
	# plt.title('amplitude')
	# plt.plot(amp1)
	# plt.subplot(3,1,2)
	# plt.title('Short term energy')
	# plt.plot(e1)

	# plt.subplot(3,1,3)
	# plt.title('FFT')
	# plt.plot(fft1)u

	# plt.show()
	print('\n''\n''\n''\n''\n')
	filename2=input('Give a voice authentication password:')
	record_audio(filename2)
	sampling_rate,signal=read_audio(filename2)
	sound = pm.Sound(filename2)
	# print(type(signal))
	# print(signal)
	print('Feature extraction started')
	time.sleep(1)
	print('Calculating amplitude...')
	time.sleep(1)
	# amp1, e1, fft1=find_features(signal,sampling_rate,frame_size,frame_stride,'hamming')
	print('calculating energy...')
	time.sleep(1)
	print('calculating FFT...')
	time.sleep(1)

	print("MFCC______________________________")
	
	time.sleep(1)
	print('Feature extraction completed')
	time.sleep(3)
	mfcc2=find_mfcc(sound)
	#print(find_mfcc(sound))
	amp2, e2, fft2=find_features(signal,sampling_rate,frame_size,frame_stride,'hamming')
	amp1=amp1.flatten()
	amp2=amp2.flatten()
	fft1=fft1.flatten()
	fft2=fft2.flatten()
	mfcc1=mfcc1.flatten()
	mfcc2=mfcc2.flatten()
	e1=[sum(x) for x in e1]
	e2=[sum(x) for x in e2]
	# fft1=np.array(fft1)
	# fft2=np.array(fft2)
	# e1=e1.flatten()
	# print('correaltio',np.corrcoef(amp1[:20],amp2[:20]))
	# print('correaltio',np.corrcoef(e1[:20],e2[:20]))
	# plt.plot(np.corrcoef(e1[:20],e2[:20]))
	# plt.show()
	ecorr=np.corrcoef(e1,e1)
	# print(amp1.shape)
	# print(amp2.shape)
	amp1,amp2=padding(amp1,amp2)
	e1,e2=padding(e1,e2)
	fft1,fft2=padding(fft1,fft2)
	mfcc1,mfcc2=padding(mfcc1,mfcc2)
	# print('after')
	# print(amp1.shape)
	# print(amp2.shape)
	#ampcorr=np.corrcoef(amp1,amp2)
	# print('FFT')
	# print(fft1.shape)  
	# print(fft2.shape)
	#fftcorr=np.correlate(fft1,fft2)
	# print(ampcorr)
	# print(fftcorr)
	# print(ecorr,"\n\n\n\n",ampcorr,'\n\n\n',fftcorr)
	d1, path = dtw(amp1, amp2, dist=euclidean)
	d2, path = dtw(e1, e2, dist=euclidean)
	d4, path = dtw(mfcc1, mfcc2, dist=euclidean)
	# print(d1)
	# print(d2)
	# print(d4)
	mfcc_corr=np.correlate(mfcc1,mfcc2)
	coor= mfcc_corr.flatten()
	print(coor[0]/10000000)
	#print(coor[1])
	if coor[0]/10000000>180:
		print('positive corelation')
	else:
		print('negative corelation')
	time.sleep(3)
	# print(coor[1])

	plt.plot(amp1)
	plt.plot(amp2)
	plt.show()

	plt.plot(e1)
	plt.plot(e2)
	plt.show()

	# plt.plot(fft1)
	# plt.plot(fft2)
	# plt.show()


	plt.plot(mfcc1)
	plt.plot(mfcc2)
	plt.show()

	#Dynamic Time Warping
	# manhattan_norm = lambda x, y: np.abs(x - y)
	# #d1, cost_matrix_amp1, acc_cost_matrix_amp2, path1 = dtw(np.asarray(amp1), np.asarray(amp2), dist=manhattan_norm)
	# d2, cost_matrix_e1, acc_cost_matrix_e2, path2 = dtw(np.asarray(e1), np.asarray(e2), dist=manhattan_norm)
	# #d3, cost_matrix_fft1, acc_cost_matrix_fft2, path3 = dtw(np.asarray(fft1), np.asarray(fft2), dist=manhattan_norm)
	# d4, cost_matrix_mfcc1, acc_cost_matrix_mfcc2, path4 = dtw(np.asarray(mfcc1), np.asarray(mfcc2), dist=manhattan_norm)
	# #print(d1)
	# print(d2)
	# #print(d3)
	# print(d4)
	d1, path = dtw(amp1, amp2, dist=euclidean)
	d2, path = dtw(e1, e2, dist=euclidean)
	d4, path = dtw(mfcc1, mfcc2, dist=euclidean)
	print(d1)
	print(d2)
	print(d4)
	# d3, path = dtw(fft1, fft2, dist=euclidean)
	# d4, path = dtw(mfcc1, mfcc2, dist=euclidean)
 	# print(d1)
	# print(d2)
	# print(d3)
 	# print(d4)

if __name__=='__main__':
		main()

