# General
My MSc thesis project - and the code used for [A semisupervised machine learning search for never-seen gravitational-wave sources](https://academic.oup.com/mnras/article/500/4/5408/5983106?guestAccessKey=580a9c7b-00e5-4463-b48e-7ea7e68a8c7d).

The code is divided into four folders - three for the three machines I used to run the code, and one shared.
The three machines, and motivation for using each, are:

- astrophys - the astrophysics department's cluster, useful because of the large storage space available and the 'bigmem' node - a node with 1000GB RAM.
- power - the university's power cluster, useful because of the high performance GPU's installed on it (NVIDIA Tesla V100 at the time).
- local - my laptop, used to run Juptyer notebooks for post-processing and visualizations.

Both astrophys and power are high performance clusters, running PBS servers. Therefore, the code run on these clusters was written as `.py` scripts. Each script (or group of scripts performing similar tasks) has a corresponding `.pbs` file, containing pbs commands to submit the script to the cluster. The submission itself is then done using the following command:
```
qsub -q QUEUE_NAME PBS_FILE.pbs
```

# Code
## Astrophys
Astrophys was used to download bulk GW strain files from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/), and to pre-process the strain data in order to generate spectrograms.
The astrophys subfolder tree:

- bulk_data_urls - this folder contains files with lists of urls to the GW strain files on GWOSC, from both O1 and O2, and from both LIGO detectors (H1 and L1).
- cnn - a folder containing a script used to feed a spectrogram through a CNN and extract the relevant features from it. This was used only for a small part of the project, when the power cluster was down (most of the CNN related processing was done on power).
- condition - scripts used to 'condition' the strain data, meaning to pre-process it and generate spectrograms. This folder contains three files:
  - combine_segment.py - a script used for combining all the conditioned spectrograms from the same segment into a single file ()
  - condition_raw.py - a script used for conditioning raw GW strain data and generating spectrograms, non-parallel version - obsolete.
  - condition_raw_par.py - a script used for conditioning raw GW strain data and generating spectrograms, parallel version (this one should be used). The conditioning is performed on short chunks, and the spectrograms from each chunk are saved to a seperate file, therefore combine_segment.py is required to combine these files into a single file for each segment (not strictly necessary, but helps avoid a very large number of small files).
- download - scripts used to download 
