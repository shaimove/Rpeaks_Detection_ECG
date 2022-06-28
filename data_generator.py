# data generator function
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Sequence
from wfdb.processing import normalize_bound
from utils import get_noise

# beacuse the data set is so small, we will read all the data at init stage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dataset_gen(Dataset):
    
    def __init__(self,
                 signals: List = None,
                 peaks: List = None,
                 labels: List = None,
                 ma = None,
                 bw = None,
                 win_size = 1000,
                 add_noise = True):

        """
        Generate ECG data with R-peak labels.

        Data generator that yields training data as batches. Every instance
        of training batch is composed as follows:
        1. Select one ECG signal from given list of ECG signals
        2. Randomly select one window of given win_size from selected signal
        3. Check that window has at least one beat and that all beats are
           labled as normal
        4. Create label window corresponding the selected window
            -beats and four samples next to beats are labeled as 1 while
             rest of the samples are labeled as 0
        5. Normalize selected signal window from -1 to 1
        6. Add noise into signal window and normalize it again to (-1, 1)
        7. Add noisy signal and its labels to trainig batch
        8. Transform training batches to arrays of needed shape and yield
           training batch with corresponding labels when needed

        Parameters
        ----------
        signals : list
            List of ECG signals
        peaks : list
            List of peaks locations for the ECG signals
        labels : list
            List of labels (peak types) for the peaks
        ma : array
            Muscle artifact signal
        bw : array
            Baseline wander signal
        win_size : int
            Number of time steps in the training window
        """
        
        self._signals = signals
        self._peaks = peaks
        self._labels = labels
        self._ma = ma
        self._bw = bw
        self._win_size = win_size
        self._add_noise = add_noise
        
    def __len__(self):
        return len(self._signals)
    
    def choose_normal_signal(self, sig, p4sig, plabels, index):
        i = 0
        while True:
            i = i + 1
            # Select one window randomly
            beg = np.random.randint(sig.shape[0]-self._win_size)
            end = beg + self._win_size

            # Select peaks that fall into selected window.
            # Buffer of 3 to the window edge is needed as labels are
            # inserted also next to point)
            ind_beg = np.searchsorted(p4sig, beg + 3, side='right')
            ind_end = np.searchsorted(p4sig, end - 3, side='left')
            p_in_win = p4sig[ind_beg:ind_end] - beg
            
            # Select labels that fall into selected window
            lab_in_win = plabels[ind_beg:ind_end]
            
            # Check that there is at least one peak in the window, and Check that every beat in the window is normal beat
            if (p_in_win.shape[0] >= 1) and np.all(lab_in_win == 1):
                return lab_in_win, p_in_win, beg, end
            
            if i > 300: # maybe no normal peaks in this ECG, generate new signal from other ECG signal
                index_new = np.random.randint(len(self._signals))
                sig = self._signals[index_new]
                p4sig = self._peaks[index_new]
                plabels = self._labels[index_new]
                i = 0
            
                
                    
    def __getitem__(self, index):
        # take signal, peaks and labels from selected index
        sig = self._signals[index]
        p4sig = self._peaks[index]
        plabels = self._labels[index]
        
        # get signal and peaks with only normal peaks
        lab_in_win, p_in_win, beg, end = self.choose_normal_signal(sig, p4sig, plabels, index)
        
        # Create labels for data window
        window_labels = np.zeros(self._win_size)
        np.put(window_labels, p_in_win, lab_in_win)
        
        # Put labels also next to peak
        np.put(window_labels, p_in_win+1, lab_in_win)
        np.put(window_labels, p_in_win+2, lab_in_win)
        np.put(window_labels, p_in_win-1, lab_in_win)
        np.put(window_labels, p_in_win-2, lab_in_win)

        # Select data for window and normalize it (-1, 1)
        data_win = normalize_bound(sig[beg:end], lb=-1, ub=1)
                                         
        # Add noise into data window and normalize it again
        if self._add_noise:
            data_win = data_win + get_noise(self._ma, self._bw, self._win_size)
            data_win = normalize_bound(data_win, lb=-1, ub=1)
        
        # convert to torch
        X = torch.from_numpy(np.asarray(data_win)).float().unsqueeze(1).to(device) 
        y = torch.from_numpy(np.asarray(window_labels)).float().unsqueeze(1).to(device) 
        
        return X,y