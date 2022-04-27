import matplotlib
matplotlib.use('Agg')
from ligotools import *
from ligotools import utils as ut
import json
import os
import numpy as np
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy import signal
import h5py


#hide warnings
import warnings
warnings.filterwarnings("ignore")

data = "data/BBH_events_v3.json"
sub_data =  json.load(open(data,"r"))
eventname = 'GW150914' 
event = sub_data[eventname]
fn_H1 = 'data/' + event['fn_H1'] 
fs = event['fs']
tevent = event['tevent']
fband = event['fband']
fn_template = 'data/' + event['fn_template']
f_template = h5py.File(fn_template, "r")
template_p, template_c = f_template["template"][...]
template = (template_p + template_c*1.j) 
NFFT = 4*fs
NOVL = NFFT/2
strain_H1, time, chan_dict_H1 = loaddata(fn_H1, 'H1')
dt = time[1] - time[0]
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
psd_H1 = interp1d(freqs, Pxx_H1)
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
deltat_sound = 2.
indxd = np.where((time >= tevent-deltat_sound) & (time < tevent+deltat_sound))
fshift = 400.
datafreq = np.fft.fftfreq(template.size)*fs
df = np.abs(datafreq[1] - datafreq[0])

def test1():
    strain_H1_whiten = ut.whiten(strain_H1,psd_H1,dt)
    assert len(strain_H1_whiten) == 131072
    assert type(strain_H1_whiten) is np.ndarray
    

def test2():
    strain_H1_whiten = ut.whiten(strain_H1,psd_H1,dt)
    strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
    ut.write_wavfile(eventname+"_H1_whitenbp.wav",int(fs), strain_H1_whitenbp[indxd])
    file = wavfile.read("audio/" + eventname + "_H1_whitenbp.wav")
    
    assert len(file) == 2
    assert len(file[1]) == 16384
    assert type(file[1]) is np.ndarray
    
    os.remove("audio/" + eventname + "_H1_whitenbp.wav")
    
def test3():
    strain_H1_whiten = ut.whiten(strain_H1,psd_H1,dt)
    strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
    strain_H1_shifted = ut.reqshift(strain_H1_whitenbp,fshift=fshift,sample_rate=fs)
    
    assert min(strain_H1_shifted) == -1375.718291963223
    assert type(strain_H1_shifted) is np.ndarray

def test4():

    psd_window = np.blackman(NFFT)
    dwindow = signal.blackman(template.size) 
    template_fft = np.fft.fft(template*dwindow) / fs
    
    data = strain_H1.copy()
    
    data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
    data_fft = np.fft.fft(data*dwindow) / fs
    power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    indmax = np.argmax(SNR)
    timemax = time[indmax]
    SNRmax = SNR[indmax]
    
    d_eff = sigma / SNRmax
    horizon = sigma/8
    phase = np.angle(SNR_complex[indmax])
    offset = (indmax-peaksample)

    template_phaseshifted = np.real(template*np.exp(1j*phase))    
    template_rolled = np.roll(template_phaseshifted,offset) / d_eff  
    
    
    template_whitened = ut.whiten(template_rolled,interp1d(freqs, data_psd),dt)
    template_match = filtfilt(bb, ab, template_whitened) / normalization 
    
    pcolor='r'
    strain_H1_whiten = ut.whiten(strain_H1,psd_H1,dt)
    strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
    strain_whitenbp = strain_H1_whitenbp
    template_H1 = template_match.copy()
    det = 'H1'
    plottype = "png"
    
    ut.plot_function(time, timemax, SNR, pcolor, det,eventname, plottype, tevent,strain_whitenbp, template_match, template_fft, datafreq, d_eff, data_psd, freqs, fs)
    
    assert os.path.exists('figurs/' + eventname+"_"+det+"_matchfreq."+plottype)
    assert os.path.exists('figurs/' + eventname+"_"+det+"_matchtime."+plottype)
    assert os.path.exists('figurs/' + eventname+"_"+det+"_SNR."+plottype)
    
    os.remove('figurs/' + eventname+"_"+det+"_matchfreq."+plottype)
    os.remove('figurs/' + eventname+"_"+det+"_matchtime."+plottype)
    os.remove('figurs/' + eventname+"_"+det+"_SNR."+plottype)
    
    
    
    