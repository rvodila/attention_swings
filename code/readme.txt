Decoding lateral attention shifts in fixed-gaze task suign two appraoches: 
    Decoding oscillatory dynamics using ALPHA band envelope.
    Decoding P300 component of time-locked responses to target stimulus.
     
Loading:

    preprocessing order in loading script:
        1.  load raw xdf
        2.  pick electrodes
        3.  remove bad channels per participant
        5.  apply band-pass filter (alpha: [0.5,30], p300: [0.5,8])
        4.  set common average reference (CAR)
        5.  epoch raw
        6.  Downsample (500Hz > 120Hz)


P300 Pipeline


    Preprocessing:
        Epo-level Decoding by fitting LDA
        Trial-level inference by correlating





Alpha Pipeline
    ../alpha_visualize_features.ipynb:
        *Visualize features as the model will see them at a set of electrodes. 
        *Features are computed as the log-transfomred Hilbert-amplitude of alpha band
        Saves one barplot features for a set of electrodes and the difference plot (left vs rigth channels) 
    ../TFR_alpha_lateralization.ipynb
        *Visualize alpha band lateralization effect in time-frequency-domain. 
        Saves one plot:
            Trial-average TFR & extracted Alpha-band amplitude of specific electrodes as a difference of right - left condition 
            for left and right electrode.
    ../alpha_subset_LDA.ipynb
        
    Preprocessing:
        Manual Channel selection
        Common Spatial Pattern for virtual channels
    Analysis:
        Fitting LDA