#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: RV
"""

import os
from os.path import join
import numpy as np
import mne
from mnelab.io.xdf import read_raw_xdf as read_raw
import pyxdf

# Set MNE log level
mne.set_log_level('warning')

# Define directories
wd = r'C:\Users\Radovan\OneDrive\Radboud\Studentships\Jordy Thielen\root'
os.chdir(wd)
data_dir = join(wd, "data")
experiment_dir = join(data_dir, "experiment")
files_dir = join(experiment_dir, 'files')
sourcedata_dir = join(experiment_dir, 'sourcedata')
derivatives_dir = join(experiment_dir, 'derivatives')
raw_dir = join(derivatives_dir, 'preprocessed', 'raw')

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
#loc_file = os.path.join(experiment_dir, 'files', "biosemi64.loc")
montage = mne.channels.read_custom_montage(join(files_dir, "biosemi64.loc"))


# Subjects and bad channel rejection mapping
subjects = ["VPpdia", "VPpdib", "VPpdic", "VPpdid", "VPpdie", "VPpdif", "VPpdig", "VPpdih", "VPpdii", "VPpdij",
            "VPpdik", "VPpdil", "VPpdim", "VPpdin", "VPpdio", "VPpdip", "VPpdiq", "VPpdir", "VPpdis", "VPpdit",
            "VPpdiu", "VPpdiv", "VPpdiw", "VPpdix", "VPpdiy", "VPpdiz", "VPpdiza", "VPpdizb", "VPpdizc"]


#Uniform sfreq for concatenation
target_sfreq = 500  

# Loop through subjects
for i_subject, subject in enumerate(subjects):
    print("-" * 25)
    print(f"Subject: {subject}")

    # Process only the 'covert' task
    task = 'covert'
    print(f"\tTask: {task}")
    task_raws = []  # List to hold raw objects for all runs in the task

    # Loop through runs for the 'covert' task
    for i_run, run in enumerate(runs.get(task, [])):  # Use .get() to handle missing task gracefully
        print(f"\t\tRun: {run}")

        # Run XDF file name
        fn = os.path.join(sourcedata_dir, f"sub-{subject}", "ses-S001", "eeg",
                              f"sub-{subject}_ses-S001_task-{task}_run-{run}_eeg.xdf")

        # Read EEG data
        streams = pyxdf.resolve_streams(fn)
        names = [stream["name"] for stream in streams]
        stream_id = streams[names.index("BioSemi")]["stream_id"]

        # Use `read_raw` to load the XDF file
        raw = read_raw(fn, stream_ids=[stream_id], verbose=False)

        raw.filter(0.1, 60)

        # Rename channels and set montage
        raw.rename_channels(channel_dict)
        raw.set_montage(montage)

        # Downsample the raw object
        raw.resample(sfreq=target_sfreq, verbose=False)

        # Append the raw object to the task's list of raw objects
        task_raws.append(raw)

    # Concatenate all runs into a single raw object for the 'covert' task

    combined_raw = mne.concatenate_raws(task_raws)

    # Save the concatenated raw data for the 'covert' task
    save_dir = os.path.join(derivatives_dir, "preprocessed", "raw", f"sub-{subject}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    combined_save_path = os.path.join(save_dir, f"sub-{subject}_task-{task}_raw.fif")
    combined_raw.save(combined_save_path, overwrite=True)