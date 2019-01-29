# Cognitive State Prediction via A Two-Dimensional Feature Vector
Implementation of the Voxel Weight-based feature generation method.

## Dataset
The dataset used to generate the VW features is from the StarPlus experiment: http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www.
It is comprised of fMRI data for six human subjects.

## Scripts
#### Quick Run
Running the script "VW_Classification.m" will generate the Voxel Weight-based features, and then train and test the features for all subjects using logistic regression classifier.

### convert_single_subject_data_to_matrix.m
This script transform the subject data to a single matrix with the rows as voxels(features) and coloumn as trials(samples)

### classify_VW_features.m
This script generates training and testing sets of VW features and classify it using leave-one-sample out cross validation, then returns the average classification accurcay for all folds.

### main_VW.m
Generates, train, and test the VW features on all the six subjects.

## Citation
