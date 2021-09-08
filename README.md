Graph-based label propagation for multimodal remote sensing data
================================================================
Info
--------
Label propagation algorithm that supports heterogeneous data and
includes homophily and/or heterophily during propagation. 
Implemented on 2 multimodal remote sensing data sets: Trento & Houston
2013. Both datasets contain a fully-overlapping hyperspectral + LiDAR image.
The algorithm outputs a label prediction for each pixel in the input
image.

Two cases are distinghuised: 
- Case 1: Images of different modalities are fully overlapping --\> homogeneous graph with one node and one edge type. This case is referred
    to as 'full overlap'.

- Case 2: Images of different modalities are *not* fully overlapping --> heterogeneous graph with 2 node types and 3 edge types. This case is referred to as 'hyperspectral' or 'lidar', depending on the type of modality that is found in the non-overlapping region.

The method contains 5 hyperparameters (kNN, ep, sig\_spe, sig\_spa and r), which are tuned via
a random search using k-fold cross validation on the training set. This
happens in 'randomSearchCV\_*dataset*.py' and
'validationRandomSearch\_*dataset*.py', where *dataset* is 'Trento' or
'Houston'. The optimal hyperparameters are then stored in
'optimal\_rs\_*dataset*.pickle' . 'main.py' runs label propagation on
the test set, using the optimal hyperparameters.

Due to scalability issues, the original test set is divided into subsets
(folds) and label prop. is run on the different subfolds to obtain a
label for each sample in the test set. The subfolds are created via the
function 'generateFoldDictionary' (imported from 'generateFolds.py')
which works as follows: the test set is split into k folds (for Trento
and Houston 6 folds is used), in a stratified fashion in order to
maintain class (dis)balances like in the original test set. Next, the
main\_LP method is run on each subfold to obtain classification of the
entire test set.

Installation
----------------------
1)  Create a virtual environment (e.g. X-modalBP) with python>=3.6
2)  Activate the virtual environment and install the following packages:
    * numpy
    * matplotlib
    * scikit-learn
    * scikit-image
    * tifffile
    * loguru
    * gdal
3)  Create a seperate directory called e.g. 'data' with subdirectories
    'Trento' and 'Houston', which contain the Trento and Houston data
    sets, respectively. The path to this directory needs to be added in
    main.py (see Usage).
    
Usage
----------------------
Complete required user-input in main.py:
1)  *dataset*: which data set to use ('Trento' or 'Houston')
2)  *choice*:  which case to run (case 1: 'full overlap', case 2:
        'hyperspectral' or 'lidar')
3)  *realDataDir*: specify path to data directory 

Run label propagation on the command line by using: `python main.py`

Author
----------------------
Catherine Taelman
For questions or suggestions: cta014@uit.no
