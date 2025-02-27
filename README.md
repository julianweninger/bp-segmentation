# BP segmentation

This project contains the scripts used for data analysis in the publications:

1. Prakash et al., 2025, in press at Nature Communications, Junctional Force Patterning drives both Positional Order and Planar Polarity in the Auditory Epithelia 
1. Weninger et al., 2025, in preparation, Force patterning drives quasi-stratification, cell flows, and spatial order in auditory epithelia.

Please cite either or both publication(s) when using this code.

## Structure

This project contains the following notebooks:

1. import_data.ipynb :
    This notebook converts raw data exported from `Tissue Analyzer` to panda DataFrames and performs useful computations.
1. heaxtic_order.ipynb : 
    This notebook explains how to calculate the (corrected) hexatic order parameter.
1. analyse_BP.ipynb :
    This notebook analyses the data from BP explant segmentations
1. analyse_BP_live_imaging.ipynb :
    This notebook analyses the live imaging data from BP
1. analyse_segment_whole_BP.ipynb :
    This notebook performs the steps towards segmentation and analyses the whole BP HC antigen staining
1. analyse_BP_3D.ipynb :
    This notebook analyses the volume segmentation of HC from BP