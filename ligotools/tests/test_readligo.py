from ligotools import *
import json
import numpy as np

#hide warnings
import warnings
warnings.filterwarnings("ignore")

def test1():
    data = "data/BBH_events_v3.json"
    sub_data =  json.load(open(data,"r"))
    eventname = 'GW150914' 
    event = sub_data[eventname]
    fn_H1 = 'data/' + event['fn_H1']  
    strain, meta, channel_dict = loaddata(fn_H1, 'H1')
    assert strain.shape==(131072,)
    assert isinstance(strain, np.ndarray)

def test2():
    data = "data/BBH_events_v3.json"
    sub_data =  json.load(open(data,"r"))
    eventname = 'GW150914' 
    event = sub_data[eventname]
    fn_L1 = 'data/' + event['fn_L1']  
    strain, meta, channel_dict = loaddata(fn_L1, 'L1')
    assert strain.shape==(131072,)
    assert isinstance(strain, np.ndarray)
    
def test3():
    hd_l1 = read_hdf5('data/L-L1_LOSC_4_V2-1126259446-32.hdf5')
    assert len(hd_l1) == 7
    assert hd_l1[0].shape == (131072,)
    
def test4():
    hd_h1 = read_hdf5('data/H-H1_LOSC_4_V2-1126259446-32.hdf5')
    assert len(hd_h1) == 7
    assert hd_h1[0].shape == (131072,)