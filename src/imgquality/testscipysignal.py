'''
Created on Nov 8, 2017

@author: loitg
'''

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from classify.common import summarize

if __name__ == '__main__':
    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    freq = np.linspace(1e3, 2e3, N)
    x = amp * np.sin(2*np.pi*freq*time)
    x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    f, t, Sxx = signal.spectrogram(x, 10, nperseg=100 ,noverlap=50)
    print summarize(f)
    print summarize(t)
    print summarize(Sxx)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    
# ((51,), [0.0, 0.25, 0.5], array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5]))
# ((1999,), [50.0, 50000.0, 99950.0], array([  5.00000000e+01,   2.00300000e+04,   4.00100000e+04,
#          5.99900000e+04,   7.99700000e+04,   9.99500000e+04]))
# ((51, 1999), [5.7652737959237349e-08, 17.532281263577083, 644.93029789309037], array([  5.76527380e-08,   2.14564435e+00,   5.28522855e+00,
#          9.86207760e+00,   1.83637980e+01,   6.44930298e+02]))