from scipy.stats.stats import pearsonr
import numpy as np
import librosa

y,sr = librosa.load('organfinale.wav') 
y1,sr1 = librosa.load('organfinale.wav')
a = np.correlate(y,y1)
print(a)
b = (np.linalg.norm(a))
print(b)

