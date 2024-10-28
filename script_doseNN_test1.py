
"""
Created on Friday 4 Octobre 2024
@author: lvdnbrz
"""

import sys
sys.path.append('/DATASERVER/MIC/GENERAL/STAFF/lvdnbrz/DATA_FROM_LIESBETH')


import mic
import mic.deepvoxnet
import os
import pydicom
import nibabel as nib
import dicom2nifti
import matplotlib.pyplot as plt

# %% DEFINE PATH PARAMETERS

verbose = True # print trainingstijden
data_path = ...