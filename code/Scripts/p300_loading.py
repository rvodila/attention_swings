#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: RV
"""

import os
import numpy as np
import mne
from mnelab.io.xdf import read_raw_xdf as read_raw
import pyxdf
from os.path import join
# Set MNE log level
mne.set_log_level('warning')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
# Define directories

wd = r'C:\Users\Radovan\OneDrive\Radboud\Studentships\Jordy Thielen\root'
os.chdir(wd)
data_dir = join(wd, "data")
ensure_dir(data_dir)
experiment_dir = join(wd, "data", "experiment")
ensure_dir(experiment_dir)
files_dir = join(experiment_dir, 'files')
ensure_dir(files_dir)
sourcedata_dir = join(experiment_dir, 'sourcedata')
derivatives_dir = join(join(experiment_dir, 'derivatives'))
#save_dir = os.path.join(derivatives_dir, "preprocessed", "p300", f"sub-{subject}")

# EEG channel mapping
channel_dict = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
    'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
    'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
    'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz',
    'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFz', 'B6': 'Fz', 'B7': 'F2', 'B8': 'F4',
    'B9': 'F6', 'B10': 'F8', 'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz',
    'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2',
    'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2'
}

# Tasks and runs
tasks = ["overt", "covert"]
runs = {"overt": ["001"], "covert": ["001", "002", "003", "004"]}


# Load electrode montage
loc_file = os.path.join(files_dir, "biosemi64.loc")
montage = mne.channels.read_custom_montage(loc_file)

pr = 60
fs = 120

# Load visual stimulus codes
V = np.load(os.path.join(files_dir, "mgold_61_6521.npz"))["codes"]
V = np.repeat(V, int(fs / pr), axis=1).astype("uint8")

# Subjects and bad channel rejection mapping
subjects = ["VPpdia", "VPpdib", "VPpdic", "VPpdid", "VPpdie", "VPpdif", "VPpdig", "VPpdih", "VPpdii", "VPpdij",
            "VPpdik", "VPpdil", "VPpdim", "VPpdin", "VPpdio", "VPpdip", "VPpdiq", "VPpdir", "VPpdis", "VPpdit",
            "VPpdiu", "VPpdiv", "VPpdiw", "VPpdix", "VPpdiy", "VPpdiz", "VPpdiza", "VPpdizb", "VPpdizc"]

subjects_channel_reject = {
    "VPpdib": ["CP2"],
    "VPpdih": ["C3"],
    "VPpdizb": ["Fz"],
    "VPpdizc": ["FC2"]
}

# Define sub-set of electrodes
picks_hubner = [
    "F7", "F3", "Fz", "F4", "F8", "FC1", "FC2", "FC5", "FC6", "FCz", "T7", "C3",
    "Cz", "C4", "T8", "CP1", "CP2", "CP5", "CP6", 'CPz',
    "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2"
]

fs = 120  # target EEG (down)sampling frequency in Hz

bandpass = (0.5, 8)  # bandpass with low and high cutoff in Hz
tmin = -1  # trial start in seconds
tmax = 21  # trial duration in seconds

# Loop through each subject
for subject in subjects:
    print("-" * 25)
    print(f"Processing subject: {subject}")

    # Loop through each task
    for task in tasks:
        print(f"\tTask: {task}")

        # Initialize data lists
        X, y, z = [], [], []

        # Loop through runs for the current task
        for run in runs[task]:
            # Construct the file path for the XDF file
            fn = os.path.join(sourcedata_dir, f"sub-{subject}", "ses-S001", "eeg",
                              f"sub-{subject}_ses-S001_task-{task}_run-{run}_eeg.xdf")

            # Read EEG data
            streams = pyxdf.resolve_streams(fn)
            names = [stream["name"] for stream in streams]
            stream_id = streams[names.index("BioSemi")]["stream_id"]
            raw = read_raw(fn, stream_ids=[stream_id], verbose=False)

            # Rename channels and set montage
            raw.rename_channels(channel_dict)
            raw.set_montage(montage)

            # De-meaning
            raw._data[0, :] -= np.median(raw._data[0, :])
            raw._data[0, :] = raw._data[0, :] > 0
            raw._data[0, :] = np.logical_and(raw._data[0, :], np.roll(raw._data[0, :], -1))

            # Find events in the trigger channel
            events = mne.find_events(raw, stim_channel="Trig1", verbose=False)

            # Handle cases where no hardware events are found
            if events.shape[0] == 0:
                print(f"\t\tNo events found in trigger channel. Falling back to marker stream.")
                streams = pyxdf.load_xdf(fn)[0]
                names = [stream["info"]["name"][0] for stream in streams]

                stream = streams[names.index("KeyboardMarkerStream")]
                t_mrk = [t for t, mrk in zip(stream["time_stamps"], stream["time_series"])
                         if mrk[2] == "start_stimulus"]

                stream = streams[names.index("BioSemi")]
                t_eeg = stream["time_stamps"]

                events = np.zeros((len(t_mrk), 3), dtype=events.dtype)
                events[:, 0] = np.array([np.argmin(np.abs(t_eeg - t)) for t in t_mrk])

            # pick sub-set (Huebner)
            raw.pick(picks_hubner)
            
            # Reject bad channel
            if subject in subjects_channel_reject:
                channels_to_drop = subjects_channel_reject[subject]
                #raw_ch_names = [str(ch) for ch in raw.info['ch_names']]
                 # Ensure that only existing channels are dropped
                channels_to_drop = [ch for ch in channels_to_drop if ch in raw.info['ch_names']]
                if channels_to_drop:
                    raw.drop_channels(channels_to_drop)

            # Spectral bandpass filter
            raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)

            # Set CAR afterwards so unused channels don't contaminate
            raw.set_eeg_reference(ref_channels='average', projection=False, verbose=None)
            
            epo = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, baseline=None,
                             preload=True, verbose=False)
            # Resampling
            epo = epo.resample(sfreq=fs, verbose=False)

            # Extract data, labels, and targets
            X.append(epo.get_data())
            streams = pyxdf.load_xdf(fn)[0]
            names = [stream["info"]["name"][0] for stream in streams]
            marker_stream = streams[names.index("KeyboardMarkerStream")]

            cued_side = np.array([marker[3].lower().strip('""') == "right"
                                  for marker in marker_stream["time_series"]
                                  if marker[2] == "cued_side"])
            left_target = np.array([x[3].split(";")[0].split("=")[1] == "hour_glass"
                                    for x in marker_stream["time_series"]
                                    if x[2] == "left_shape_stim"]).reshape((cued_side.size, -1))
            right_target = np.array([x[3].split(";")[0].split("=")[1] == "hour_glass"
                                     for x in marker_stream["time_series"]
                                     if x[2] == "right_shape_stim"]).reshape((cued_side.size, -1))

            targets = np.stack((left_target, right_target), axis=2)
            y.append(cued_side)
            z.append(targets)

        # Stack data for the current task
        X = np.concatenate(X, axis=0).astype("float32")
        y = np.concatenate(y, axis=0).astype("uint8")
        z = np.concatenate(z, axis=0).astype("uint8")

        # Save processed data
        save_dir = os.path.join(derivatives_dir, "preprocessed", "p300", f"sub-{subject}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir, f"sub-{subject}_task-{task}_p300.npz"), X=X, y=y, z=z, V=V, fs=fs)
