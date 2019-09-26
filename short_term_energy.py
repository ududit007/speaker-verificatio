import numpy 
from numpy import linspace
import matplotlib.pyplot as plt 

def short_term_energy(frames):
	STEs=[]
	for frame in frames:
		STE=frame**2
		STEs.append(STE)
	return STEs