# General
My MSc thesis project - and the code used for [A semisupervised machine learning search for never-seen gravitational-wave sources](https://academic.oup.com/mnras/article/500/4/5408/5983106?guestAccessKey=580a9c7b-00e5-4463-b48e-7ea7e68a8c7d).

The code is divided into four folders - three for the three machines I used to run the code, and one shared.
The three machines, and motivation for using each, are:

- astrophys - the astrophysics department's cluster, useful because of the large storage space available and the 'bigmem' node - a node with 1000GB RAM.
- power - the university's power cluster, useful because of the high performance GPUs installed on it (NVIDIA Tesla V100 at the time).
- local - my laptop, used to run Juptyer notebooks for post-processing and visualizations.

Both astrophys and power are high performance clusters, running PBS servers. Therefore, the code run on these clusters was written as `.py` scripts. Each script (or group of scripts performing similar tasks) has a corresponding `.pbs` file, containing pbs commands to submit the script to the cluster. The submission itself is then done using the following command:
```
qsub -q QUEUE_NAME PBS_FILE.pbs
```

# Project pipeline
## Training
### On astrophys:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Download files from GWOSC | download GW strain files containing glithces labeled by Gravity Spy | `download_gs.py` | `pbs_download.pbs` | in the PBS file uncomment/comment out the relevant lines. |
| Generate training set | generate spectrograms of the time-stamps in Gravity Spy and label them accordingly | `create_gs_fromraw.py` | `pbs_create_fromraw.pbs` | in order to avoid issues on the cluster, the script conditions only 1000 time-stamps at a time, so the script should be run several times, with a different value for the  `idx_start` variable, starting from 0 and changing in steps of 1000 (the number of time-stamps conditioned each time can be increased somewhat, to around 2500 but not much more as the script is written currently). |
| Combine training set | combine the files generated in the previous step to a single `.hdf5` containing the entire training set | `combine_fromraw.py` | `pbs_combine_fromraw.pbs` | |
| Transfer to power | transfer the `.hdf5` file containing the training set to power cluster | | | can be done using the `scp` command:<br>`scp FILE PATH_ON_POWER` |

### On power:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Split and augment | split the data set into training, test and validation sets, and augment the training set | `split_augment.py` | `pbs_split_augment.pbs` | |
| Train model | train a deep CNN using the generated spectrograms | `train_network.py` | `pbs_train_network.pbs` | several training hyper-parameters can be chosen in the parameters file `gsparams.py` such as the pre-trained model to use (`extract_model` variabel), the input size, the optimization method and the name given to thel model. |
| Evaluate model | evaluate the trained model and print the loss and classification accuracy for the training, test and validation sets | `model_evaluation.py` | `pbs_model_eval.pbs` | prints the results to `stdout`. |
| Extract labeled set features | feed the labeled data set through the network and extract feature space representations, softmax predictions and values relevant to the Gram matrix method | `extract_gram.py` | `pbs_extract.pbs` | in the PBS file uncomment/comment out the relevant lines. |

## Search
### On astrophys:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Download files from GWOSC | download bulk GW strain files | `download_16.py` | `pbs_download.pbs` | in the PBS file uncomment/comment out the relevant lines. in the script choose the desired detector (currently either 'H1' or 'L1') and the desired file indices. |
| Condition data | condition the raw strain data and generate spectrograms. | `condition_raw_par.py` | `pbs_condition.pbs` | in the script, choose the detector to be conditioned (either 'H' or 'L') and the segment numbers to be conditioned (when I ran this script I ran it on groups of 50 segments at a time). This script generates a folder for each segment, and within it a seperate `.hdf5` file for each chunk conditioned. |
| Combine spectrograms | combine the spectrograms generated in each segment to a single `.hdf5` file | `combine_segment.py` | `pbs_combine.pbs` | as in the previous step, choose the desired detector and segments to be combined. |
| Transfer to power | transfer the `.hdf5` files containing the spectrograms to be processed with the network to power cluster | | | can be done using the `scp` command:<br>`scp FILE PATH_ON_POWER`. Since there might be many large spectrogram files, I did this in groups, and used the public storage folders on power, /scratch100/, /scratch200/ or /scratch300/ to temporarily store these files only when processing them through the network. |

### On power:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Extract features | process the unlabeled spectrograms through the network and extract the feature space representations, softmax predictions and Gram method deviations. | `extract_gram_unlabeled.py` | `pbs_extract.pbs` | in the PBS file uncomment/comment out the relevant lines. In the script choose the desired detector. |
| Transfer to local | transfer the `.hdf5` files containing the extracted features for both training set as well as the unlabeled spectrograms to be searched to the machine that will run the jupyter notebooks in the 'local' subfolder. | | | can be done the `scp` command |

### On local:
| Action | Description | Notebook to run | Notes |
| ------ | ----------- | --------------- | ----- |
| Generate UMAP | generate the UMAP mapping between the feature space and the map space, and save it to a file. The mapping is generated using the training set features. | `compute_umap.ipynb` | the notebook also enables to generate an interactive plot of th map space and explore it. |
| Generate examples | generate file containing examples of the different glitch classes in the GS data set. | `create_examples.ipynb` | optional, these examples can be used in some of the visualizations of the map space. |
| Estimate distribution | estimate the map space distribution using kernel density estimation on the training set, and save the estimator. | `compute_kde.ipynb` | this notebook enables to generate contours of density equal to the threshold. The threshold can be determined later using the unlabeled data, so the this notebook can be used again after the threshold is chosen to generate the contours. |
| Map spectrograms | map unlabeled spectrograms to the map space, and append them to the features files. | `create_maps.ipynb` | choose the path to the folder containing the desired features files (the same notebook is later used to map the injected files, and the path currently defined is the path to the injection files). |
| Flag outliers | flag time-stamps containing outliers according to the different methods, and save them to `tois.hdf5`. | `flag_outliers.ipynb` | this notebook can be used to determine the thresholds of the density, and Gram, methods - the thresholds are chosen to achieve a desired number of identified outliers. |
| Plot map space | generate a plot of the map space using matplotlib | `plot_map_space.ipynb` | optional, this notebook was used to generate the plot in the paper. | Transfer `tois.hdf5` to astrophys | transfer the outlier time-stamps file to astrophys. | | the file is saved to the github folder (to the 'shared' subfolder), so this can be done simply by using the `git pull` command on astrophys. |

### On astrophys:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Get flagged | generate a file containing the spectrograms of the outlier time-stamps for each outlier detection method. | `get_pois_script.py` | `pbs_pois.pbs` | the script should be run several times, once for each outlier detection method, and for each detector (chosen by the `method` and `detector` variables in the script). |
| Transfer to local | transfer the files containing the outlier spectrograms to the machine running the jupyter notebooks. | | | |

### On local:
| Action | Description | Notebook to run | Notes |
| ------ | ----------- | --------------- | ----- |
| Visualize outliers | generate plots of the outlier spectrograms, in order to identify the types of outliers. | `plot_flagged.ipynb` | this notebook is not organized and documented well, it probably should be written from scratch. |

## Evaluation
### On local:
| Action | Description | Notebook to run | Notes |
| ------ | ----------- | --------------- | ----- |
| Generate injection parameters | generate `.csv` files containig the random injection times (with and without the no glitch constraint) and the random sky locations (and polarization parameters). | `inject_times_gen.ipynb` | I forgot to add this notebook to the git at first, so it's not documented very well and might not work currently (but the `.csv` were already generated and are in the 'shared' subfolder). |
| Generate white noise waveform | generate the random white noise waveform to be injected. | `gen_wn.ipynb` |  |
| Transfer to astrophys | transfer the `.csv` files containing injection times and sky locations to astrophys. | | the files are saved to the github folder (to the 'shared' subfolder), so this can be done simply by using the `git pull` command on astrophys. |

### On astrophys:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Inject ad-hoc waveforms | inject ad-hoc waveforms, to a small number of time-stamps with the no glitch constraint. | `inject_times.py` | `pbs_inject.pbs` | in the PBS file uncomment/comment out the relevant lines. This script should be run once for each detector and for each waveform type (chosen by the `detector` and `inj_type` variables). The script is not written very well, so in addition to the variables, the loop related to the desired injection type should be uncommented, and the rest commented out. |
| Inject CCSNe waveforms | inject CCSNe waveforms, to a small number of time-stamps with the no glitch constraint. | `inject_ccsn.py` | `pbs_inject.pbs` | in the PBS file uncomment/comment out the relevant lines. This script should be run once for each detector and for each CCSNe paper (chosen by the `detector` and `ccsn_paper` variables). |
| Transfer to power | transfer the `.hdf5` files containing the injected spectrograms to be processed with the network to power cluster | | | can be done using the `scp` command:<br>`scp FILE PATH_ON_POWER`. |

### On power:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Extract features | process the injected spectrograms through the network and extract the feature space representations, softmax predictions and Gram method deviations. | `extract_gram_unlabeled.py` | `pbs_extract.pbs` | in the PBS file uncomment/comment out the relevant lines. In the script choose the desired detector. |
| Transfer to local | transfer the `.hdf5` files containing the extracted features machine that runs the jupyter notebooks in the 'local' subfolder. | | | can be done the `scp` command |

### On local:
| Action | Description | Notebook to run | Notes |
| ------ | ----------- | --------------- | ----- |
| Map spectrograms | map injected spectrograms to the map space, and append them to the features files. | `create_maps.ipynb` | choose the path to the folder containing the desired features files. |
| Process injected | post-process the injected spectrograms (both the ad-hoc and CCSNe waveforms), and get detection statistics using the different outlier detection methods. | `process_injected.ipynb` and `process_injected_ccsn.ipynb` | these notebook is not organized and documented well, theyt probably should be written from scratch. |

### On astrophys:
| Action | Description | Script to submit | PBS file | Notes |
| ------ | ----------- | ---------------- | -------- | ----- |
| Gather statistics | for the chosen waveforms, inject into additional time-stamps (without the no glitch constraint) for additional detection statistics. The waveforms chosen are the waveforms for which at least half of the injections were detected in both detectors by at least one method. | `inject_stats.py` and `inject_ccsn_stats.py` | `pbs_inject.pbs` | the same notes as for the previous injection scripts apply. In addition, since these scripts generate 1000 injections, they are divided into groups of 100 injections (which generate separate files for each group). |
| Combine injections | combine the injections generated by the previous scripts into a single file for each injection waveform. | `combine_injected.py` | `pbs_combine_injected.pbs` | should be performed for each injected waveform separately. |

The final steps as for the previous injections should be repeated for the new injections (extract features using the network on power, map the features to the map space and post-process the injections using the relevant notebooks on local).

# Project subfolder tree
A description of the project's subfolder tree.
## Astrophys
Astrophys was used to download bulk GW strain files from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/), and to pre-process the strain data in order to generate spectrograms.
The `astrophys/` subfolder tree:
- `bulk_data_urls/` - this folder contains files with lists of urls to the GW strain files on GWOSC, from both O1 and O2, and from both LIGO detectors (H1 and L1).
- `cnn/` - a folder containing a script used to feed a spectrogram through a CNN and extract the relevant features from it. This was used only for a small part of the project, when the power cluster was down (most of the CNN related processing was done on power).
- `condition/` - scripts used to 'condition' the strain data, meaning to pre-process it and generate spectrograms. This folder contains three files:
  - `combine_segment.py` - a script used for combining all the conditioned spectrograms from the same segment into a single file ()
  - `condition_raw.py` - a script used for conditioning raw GW strain data and generating spectrograms, non-parallel version - obsolete.
  - `condition_raw_par.py` - a script used for conditioning raw GW strain data and generating spectrograms, parallel version (this one should be used). The conditioning is performed on short chunks, and the spectrograms from each chunk are saved to a seperate file, therefore `combine_segment.py` is required to combine these files into a single file for each segment (not strictly necessary, but helps avoid a very large number of small files).
- `download/` - scripts used to download raw GW strain files from GWOSC. This folder contains the following files:
  - `download_16.py` - script used to download bulk 16KHz strain files.
  - `download_gs.py` - script used to download files containing glithces from the Gravity Spy data set.
  - `files_with_glitch.txt` - list of the urls of files containing the GS data set glitches.
- `get_flagged/` - contains `get_pois_script.py`, script used to extract the spectrograms of the time-stamps flagged as outliers by the different methods, and to save them to a single file.
- `gravityspy/` - this folder contains scripts used to generate spectrograms of the time-stamps from the GS data set, and to label them according to the data set. these are used as the training data set when training the CNN. This folder contains the following files:
  - `combine_fromraw.py` - a script used to combine the spectrogram files generated using `create_gs_fromraw.py` into a single file containing all the training spectrograms.
  - `create_gs_fromraw.py` - a script used to generate the spectrograms of the GS data set. The data set is too large to generate all of them in a single run of the script (as it is written, this can probably be improved), therefore it is divided to several runs, each generating a seperate file. These files are finally combined to a single file using `combine_fromraw.py`.
  - `find_missing.py` - a script to identify time-stamps from the GS data set that are not publicly available (and therefore do not exist in our data set).
  - `missing.csv` - a file containing the missing time-stamps, generated by `find_missing.py`
  - `trainingset_v1d1_metadata.csv` - the GS data set metadata file (used to find time-stamps containing the GS glitches, and to label them accordingly).
- `injections/` - this folder contains scripts used to inject waveforms into the strain data, in order to evaluate the sensitivity of the search. This folder contains the following files:
  - `combine_injected.py` - was used to combine injection files, not really required anymore.
  - `inject_ccsn.py` - scirpt used to inject simulated CCSNe waveforms (into 15 time-stamps, selected with the constraint they are classified as 'No Glitch').
  - `inject_ccsn_backup.py` - backup of an obsolete version of the previous script.
  - `inject_ccsn_stats.py` - script used to inject simulated CCSNe waveforms, in order to gather detection statistics (here, the waveforms are injected into 1000 time-stamp, randomly chosen with no constraint).
  - `inject_stats.py` - script used to inject ad-hoc waveforms, in order to gather detection statistics (here, the waveforms are injected into 1000 time-stamp, randomly chosen with no constraint).
  - `inject_times.py` - scirpt used to inject ad-hoc waveforms (into 15 time-stamps, selected with the constraint they are classified as 'No Glitch').
  - `inject_times_backup.py` - backup of an obsolete version of the previous script.
- `pbs_files/` - folder containing the `.pbs` files used to submit the `.py` scripts to the cluster. They are divided into subfolders according to the queue the scripts should be submitted to.
  - `bigmem/`
    - `pbs_combine.pbs` - used to submit `/condition/combine_segment.py`.
    - `pbs_combine_fromraw.pbs` - used to submit `/gravityspy/combine_fromraw.py`.
    - `pbs_combine_injected.pbs` - used to submit `/injections/combine_injected.py`.
    - `pbs_condition.pbs` - used to submit `/condition/condition_raw_par.py`.
    - `pbs_create_fromraw.pbs` - used to submit `/gravityspy/create_gs_fromraw.py`.
    - `pbs_inject.pbs` - used to submit `/injections/inject_*.py`.
    - `pbs_missing.pbs` - used to submit `/gravityspy/find_missing.py`.
    - `pbs_pois.pbs` - used to submit `/get_flagged/get_pois_script.py`.
  - `gpuq/`
    - `pbs_extract.pbs` - used to submit `/cnn/extract_gram_unlabeled.py`.
  - `workq/`
    - `pbs_download.pbs` - used to submit `/dowbload/download_16.py` and `/download/download_gs.py`.
- `segment_files/` - folder containing files with lists of the time segments from O1 and O2 - files with segments of available data for each detector (e.g., O1_H1_DATA), as well as files with segments for which data from both detectors are available (e.g., O1_BOTH_DATA).
- `tools/` - folder containing tools and parameter files:
  - `params.py` - parameters relevant to to conditioning phase.
  - `tools_gs.py` - tools used to condition the strain data and generate the spectrograms, non-parallel version - obsolete.
  - `tools_gs_par.py` - tools used to condition the strain data and generate the spectrograms, parallel version (this one should be used).

## Local
The `local/` subfolder tree:
- `notebooks/` - jupyter notebooks used for small processing tasks and for creating visualizations. This folder contains the following files:
  - `ccsn_waveforms.ipynb` - used for experimentation and familiarization with the simulated CCSNe waveforms, and to figure out how to inject them. Not really a part of the pipeline.
  - `compute_kde.ipynb` - used to compute and save the kde estimator (density estimator of the map space) and the contours of the threshold density.
  - `create_examples.ipynb` - create a file containing a small, random set of examples from the training set, including the spectrograms, labels, time-stamps and features extracted using the CNN.
  - `create_maps.ipynb` - for a set of files containing features exctracted using the CNN, compute the map space representations (using the mapper computed in create_umap.ipynb), and append them to each file.
  - `create_umap.ipynb` - compute and save the UMAP mapper, mapping the feature space to the map space, using the training set's extracted features.
  - `flag_outliers.ipynb` - flag outliers according to the different methods and thresholds and save the outlier time-stamps to a file.
  - `gen_wn.ipynb - generate` and save white noise waveforms that are to be injected in the evaluation phase of the pipeline.
  - `get_hardware_inj_time.ipynb` - find and save hardware injection times to a file. Not really part of the pipeline, was used to visualize the map space representation of the hardware injections.
  - `injection_waveforms.ipynb` - used for experimentation and familiarization with ad-hoc waveforms, and to figure out how to inject them (similar to the simulated `ccsn_waveforms.ipynb`). Not really a part of the pipeline.
  - `plot_flagged.ipynb` - post-processing and visualization of the outliers flagged by the different methods.
  - `plot_map_space.ipynb` - used to generate the map space figure used in the paper.
  - `process_injected.ipynb` - post-processing of the injected ad-hoc waveforms, and gather detection statistics.
  - `process_injected_ccsn.ipynb` - same as previous, but for the simulated CCSNe waveforms.
- `tools/` - tools used locally for post-processing and visualizations. This folder contains the following files:
  - Three `.txt` injections files - containing hardware injection times for both LIGO detectors and for both O1 and O2.
  - `contours_kde.sav` - data file containing the contours of threshold density in the map space.
  - `kde_0_3.sav` - data file containing the kde estimator of the map space distribution.
  - `map_tools.py` - tools used for accessing data, and some manipulations, mostly obsolete.
  - `mapper.sav` - data file containing the UMAP mapper, mapping the feature space to the map space.
  - `params.py` - parameters used by the local jupyter notebooks.
  - `plot_tools.py` - tools used to generate the interactive plots of the map space together with spectrograms of the chosen points. These plots are generated using holoviews with a bokeh backend.
  - `plot_tools_mpl.py` - tools used to generate plots that were used in the paper. These plots were generated using matplotlib (mpl allows for better/easier customization required for the paper plots).

## Power
Power was used to train the deep CNN used in the project, and to process the unlabeled spectrograms through it once it was trained. The reason is the high performance GPU installed on it reduced the training and processing time significantly.
The `power/` subfolder tree:
- `cnn/` - the scripts for training the network and processing the spectrograms with it. This folder contains the following files:
  - `extract_gram.py` - script to feed the labeled spectrograms through the trained network, and to extract the feature space representations (the penultimate layer activations) and the label predictions. In addition, this script computes the relevant values for the Gram matrix method: The minimum and maximum Gram matrix values for the training set, and the deviations for the test and validation set.
  - `extract_gram_unlabeled.py` - similar to the previous script, only this time for unlabeled spectrograms (from a detector defined in the script).
  - `model_evaluation.py` - script to evaluate the trained networks, prints losses and classification accuracies for the test, validation and training set.
  - `train_network.py` - script to define, train and save the network.
- `pbs_files/` - folder containing the `.pbs` files used to submit the `.py` scripts to the cluster. They are divided into subfolders according to the queue the scripts should be submitted to (similar to the pbs_files folder on astrophys).
  - `gpu/`
    - `pbs_extract.pbs` - used to submit `/cnn/extract_gram.py` and `/cnn/extract_gram_unlabeled.py`.
    - `pbs_model_eval.pbs` - used to submit `/cnn/model_evaluation.py`.
    - `pbs_train_network.pbs` - used to submit `/cnn/train_network.py`.
  - `workq/`
    - `pbs_split_augment.pbs` - used to submit `/split_augment/split_augment.py`.
- `split_augment/` - contains `split_augment.py`, script used to split the labeled data set into training test and validation sets, and to augment the training set.
- `tools/` - tools and parameters used by the scripts on power cluster. 
  - `gsparams.py` - parameters file, containing parameters relating to the CNN.
  - `gstools.py` - tools relating to training the CNN and to extracting the relevant features from it.

## Shared
The `shared/` folder contains files used in multiple locations. In practice, they are used only in astrophys and locally.
The `shared/` subfolder tree:
- `ccsn_wfs/` - folder containing simulated CCSNe waveforms, used to evaluate the search by injecting them into the strain data.
- `injection_params/` - folder containing several files that relate to the injection process:
  - Files ending with `params_csv.csv` - detail the injection parameters of the different waveforms to be injected.
  - Files containing `ood` or `pred` - describe injected waveforms' detection rates.
  - `sky_loc_csv.csv` - file containing randomly generated source parameters for the injected waveforms - sky location (ra, dec), polarization angle (pol) and total polarization (alpha).
  - `soft_inj_time.csv` - file containing injection times, generated randomly with the constraint that these times contain no glitch. This file contains times for both H and L detectors, however in practice, only the H generated times were used, and the L times were computed according to the sky location.
  - `soft_inj_time_no_contstraint.csv` - file containing randomly generated injection times, this time with no constraint. As with the previous files, only the H times were used, and the L times were computed according to the sky location.
  - Two `.hdf5` files - containing white noise waveforms (with two sets of parameters) to be injected.
- Three `.csv` files - containing lists of the outliers detected by each method, the detector in which each outlier occured and their types.
- `inject_tools.py` - tools used for injecting the simulated waveforms.
- `inject_tools_backup.py` - backup of the previous file - obsolete version.
- `tois.hdf5` - file containing the 'times of interest' - the outlier time-stamps for each method.

# Data
A description of the data generated/used in this project, and their whereabouts.
## On astrophys
The data on the astrophys cluster is stored on the `/arch/` drive - a large (175TB) storage drive.
The path to the data folder is: `/arch/tommaria/data/`, the data folder's subfolder tree:
- `bulk_data/16KHZ/` - bulk GW strain files downloaded from GWOSC (sampled at 16KHz). This folder is divided to two subfolders, one for each detector (H1/ and L1/).
- `conditioned_data/16KHZ` - files containing spectrograms generated from the raw data. This folder is divided to two subfolders, one for each detector (H1/ and L1/), and within each, to additional subfolders:
  - `combined/` - a folder containing the spectrograms generated during the search phase, divided into the following:
    - `moved/` - spectrograms that were already processed through the network during this project ('moved' to power cluster).
    - `pois/` - files containing spectrograms that were flagged as outliers by the different methods.
    - `tomove/` - generated spectrograms that were not processed in this project (they were generated after the spectrograms in `moved/` were processed, with the purpose of processing them as well, but I never got to it).
  - `injected/` - a folder containing the spectrograms of the injected waveforms, generated during the evaluation phase of the project.
- `gravityspy/` - a folder containing the labeled spectrograms used for training the network.
- `multi_scale/conditioned_data/16KHZ` - a folder containing multi-scale spectrograms, generated during a small followup of this project, a multi-scale version of the pipeline (not relevant to this project).

The `/arch/` drive is a slower, archive drive. It is used to store the bulk/generated data because of it's large volume, but when I ran the project, the files were saved to the faster `/storage/fast/` drive, in the path `/storage/fast/users/tommaria/data/` (which contains a subfolder tree very similar to the one in `/arch/tommaria/data/`).

## On power
The data on the power cluster is stored in the `/dovilabfs/` drive - Dovi's storage drive on the cluster that we purchased from the university for this project.
The path to the data relevant to this project is `/dovilabfs/work/tommaria/gw/`. The files in this directory are somewhat disorganized, and when I wrote this readme I tried to make the directories as clean as possible, but I didn't want to move around too many things because the code in the scripts described above use some of these paths. Therefore, I kept the files used by the project in the same place but tried to delete most of the files that aren't related (mostly files created during the different things I tried during the research). Some files I didn't want to delete 'just in case', so I moved them to folders named `unsorted/`. The folder's subfolder tree:
- `data/` - folder containing data used/created on power. This folder contains the following subfolders:
  - `gravityspy/` - folder containing data relevant to this project, contains the following subfolders:
    - `fromraw/` - spectrograms used for training the network. This folder contains three files, one with the full original data set generated on astrophys (ends with `gs.hdf5`), one with the augmentation spectrograms (has the suffix `augmentation3`) and one with the original data set split into training, test and validation set (has the suffix `split`).
    - `features/fromraw_gs_wrap_no_ty/new/resnet152v2/adadelta/gpu_test_15/gram/` - folder containing the files generated by processing spectrograms through the trained network and extracting the relevant features. Contains a single file - extracted features of the labeled data set, and two folders, one for each detector (`H1/` and `L1/`), containing the features extracted from the bulk unlabeled spectrograms (in the subfolder `segments/`) and the features of the injected spectrograms (in the subfolder `injected`/). The `injected/` subfolder is further divided into the different injection waveforms, and contains the `no_constraints/` subfolder that contains the waveforms injected without the 'no glitch' constraint.
  - `multi_scale/` - files relevant the 'multi-scale' followup, not relevant to this project.
