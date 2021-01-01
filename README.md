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
Astrophys was used to 
