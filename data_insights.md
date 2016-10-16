# Data Insights

## General
- Input is the decays signature of a particule collision
- Ouput is a classification on whether the particle is a Higg's Boson or not
- There are 30 features in total
- All variables are floating point except PRI_jet_num which is an integer
- Variables prefixed with DER are quantites computed from primitive features
- -999 represents values that cannot be computed or are meaningless
- training set is comprised of 250'000 events

## Possible plots
- Range of each feature
- Count number of -999 per feature.
- Rank attributes per correlation with output feature

## Ideas for feature engineering
- Bin separation of data to see its distribution
- Convert features with -999 into binary classification
