import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import scipy.ndimage

import sys
sys.path.append('/uz/data/radiotherapie/Shared/Data/Liesbeth_Vdw/phd_projects/general/dicomTONifty')
from DICOMtoNIFTI.Read_RTDose import RTDoseVolume

openbeam_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/F_phantom_fluence_fluenceprofile_openbeam/'
openbeam_dose = os.path.join(openbeam_path, 'RTDOSE.dcm')
nifti_generator = RTDoseVolume(openbeam_dose)
openbeamdose_arr = nifti_generator.get_data()
openbeamdose_aff = nifti_generator.affine

offset_x = (165 - openbeamdose_arr.shape[0]) // 2
offset_z = (165 - openbeamdose_arr.shape[1]) // 2
openbeamdose_arr_reshaped = np.zeros((165, openbeamdose_arr.shape[1], 165))
openbeamdose_arr_reshaped[offset_x:offset_x+openbeamdose_arr.shape[0],:,offset_z:offset_z+openbeamdose_arr.shape[2]] = openbeamdose_arr

slice_thickness = openbeamdose_aff[1,1]
nifti_generator.SaveAsNifti(openbeam_path, openbeamdose_arr_reshaped, openbeamdose_aff)

fm = '/uz/data/radiotherapie/Shared/Data/LoesVdb/data_preproc_initial/train/0057_DC/FL_maps_rotated.nii.gz'
fm_arr = nib.load(fm).get_fdata()
fm_aff = nib.load(fm).affine

name_beams = ['LAO30', 'RAO345', 'RAO320', 'RAO295', 'Re270', 'RPO245', 'RPO220', 'RPO195', 'LPO150'] # logische volgorde qua plaatsing rond patient
gantry_angles = [30, 345, 320, 295, 270, 245, 220, 195, 150]

fluence_volume = np.zeros((165,openbeamdose_arr.shape[1],165))

def zoom(img, zoom_factor):

    # working with square images, so normally shape[0] = shape[1]
    out = scipy.ndimage.zoom(img, zoom_factor)
    
    offset = (out.shape[0] - img.shape[0]) // 2
    out_correct_shape = out[offset:offset+img.shape[0], offset:offset+img.shape[1]]

    return out_correct_shape

for i in range(fm_arr.shape[2]):

    fluence_2D = fm_arr[:,:,i]
    fluence_3D = np.zeros((165,openbeamdose_arr.shape[1], 165))
    fluence_3D[:,0,:] = fluence_2D

    # extend fluence map according to beam divergence
    zoom_factor = 1 + (148 / 100) / openbeamdose_arr.shape[1]
    for y in range(1,openbeamdose_arr.shape[1]):
        fluence_2D_last = fluence_3D[:,y-1,:]
        fluence_3D[:,y,:] = zoom(fluence_2D_last, zoom_factor)

    new_aff = np.eye(4)
    new_aff[0,0] = fm_aff[0,0]
    new_aff[1,1] = slice_thickness
    new_aff[2,2] = fm_aff[1,1]
    new_aff[0,3] = fm_aff[0,3]
    new_aff[1,3] = fm_aff[2,3]
    new_aff[2,3] = fm_aff[1,3]

    # take attenuation into account
    fluence_3D = fluence_3D * openbeamdose_arr_reshaped

    save_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/0057_DC_FluencePred2_Lung_R2/' + 'FL' + name_beams[i] + '_volume.nii.gz'
    nib.save(nib.Nifti1Image(fluence_3D, new_aff), save_path)

    # rotate according to gantry angle
    save_path_rotated = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/0057_DC_FluencePred2_Lung_R2/' + 'FL' + name_beams[i] + '_volume_rotated.nii.gz'
    rotated_fluence_3D_arr = scipy.ndimage.rotate(fluence_3D, angle = gantry_angles[i], axes = (0,1), reshape = False)
    nib.save(nib.Nifti1Image(rotated_fluence_3D_arr, new_aff), save_path_rotated)

    fluence_volume += rotated_fluence_3D_arr

save_path_volume = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/0057_DC_FluencePred2_Lung_R2/' + 'FL_maps_volume.nii.gz'
nib.save(nib.Nifti1Image(fluence_volume, new_aff), save_path_volume)