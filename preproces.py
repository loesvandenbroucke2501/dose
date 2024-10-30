import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from scipy.signal import fftconvolve

import sys
sys.path.append('/uz/data/radiotherapie/Shared/Data/Liesbeth_Vdw/phd_projects/general/dicomTONifty')
from DICOMtoNIFTI.Read_RTDose import RTDoseVolume
from DICOMtoNIFTI.reformat import Reformatter3DTo3D
from DICOMtoNIFTI.Read_Fluencetxt import FluenceHeader
from DICOMtoNIFTI.preprocessing_gantryrotation import ImageRotator2D
from DICOMtoNIFTI.preprocessing import ImageResampling
from DICOMtoNIFTI.preprocessing_gantryrotation import ImageRotator


preproc_data_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb'
preproc_data_path_initial = os.path.join(preproc_data_path, 'data_preproc_initial')
preproc_data_path_initial_train = os.path.join(preproc_data_path_initial, 'train')

exporteclipse_data_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb'
exporteclipse_data_path_initial = os.path.join(exporteclipse_data_path, 'export_initial_nonclinicaleclipse')
exporteclipse_data_path_initial_train = os.path.join(exporteclipse_data_path_initial, 'train')

input_size = [160, 160, 160] # determines the input size of the CNN

name_beams = ['LAO30', 'RAO345', 'RAO320', 'RAO295', 'Re270', 'RPO245', 'RPO220', 'RPO195', 'LPO150'] # logische volgorde qua plaatsing rond patient
gantry_angles = [30, 345, 320, 295, 270, 245, 220, 195, 150]

# %% DOSE KERNEL

'''
dose_arr = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/F_phantom_fluence_profile_lvdnbrz_test', 'RTDOSE.nii.gz')).get_fdata()
dose_aff = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/F_phantom_fluence_profile_lvdnbrz_test', 'RTDOSE.nii.gz')).affine

field_size = 20 # mm
pixdim = 2.5 # mm
center_x = dose_arr.shape[0] // 2
center_z = dose_arr.shape[2] // 2
nb_pixels = int(field_size / pixdim) + 10 # 10 is margin for beam divergence

dose_kernel = dose_arr[center_x - nb_pixels//2:center_x + nb_pixels//2, :, center_z - nb_pixels//2:center_z + nb_pixels//2]
nib.save(nib.Nifti1Image(dose_kernel, dose_aff), os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/dose_kernel', 'dose_kernel.nii.gz'))
'''

dose_kernel = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/dose_kernel', 'dose_kernel.nii.gz')).get_fdata()
dose_aff = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/dose_kernel', 'dose_kernel.nii.gz')).affine

# %% CREATE FLUENCE VOLUME

# create fluence volume
for patient in os.listdir(preproc_data_path_initial_train):
    print(patient)

    fl = os.path.join(preproc_data_path_initial_train, patient, 'FL_maps_rotated.nii.gz')
    fl_arr = nib.load(fl).get_fdata()
    fl_aff = nib.load(fl).affine

    fl_volume_arr = np.zeros((fl_arr.shape[0],dose_kernel.shape[1],fl_arr.shape[1])) # initialize 
    slice_thickness = dose_aff[1,1]

    for i in range(fl_arr.shape[2]):

        fluence_2D = fl_arr[:,:,i]

        header = FluenceHeader(os.path.join(exporteclipse_data_path_initial_train, patient, 'FL_' + name_beams[i] + '_header.txt'))
        isocenterPosition = header.iso
        collimatorAngle = header.colAngle

        # fluence map roteren volgens collimator angle
        FluenceRotator2D = ImageRotator2D(fluence_2D, fl_aff, collimatorAngle, isocenterPosition)
        fl_arr_rotated, fl_aff_rotated = FluenceRotator2D.doInterpolation(1)
        # resamplen zodat alle fluence mappen dezelfde resolutie hebben
        FluenceResampler = ImageResampling(fl_arr_rotated)
        fl_arr_rotated, fl_aff_rotated = FluenceResampler.transform([fl_aff_rotated[0,0], fl_aff_rotated[1,1]], [2.5, 2.5], fl_aff_rotated, 1)
        # bijknippen, want resampling maakt de fluence groter 
        offset_x = int(np.round((fl_arr_rotated.shape[0] - 165) / 2))
        offset_y = int(np.round((fl_arr_rotated.shape[1] - 165) / 2))
        fl_arr_rotated = fl_arr_rotated[offset_x:offset_x + 165, offset_y:offset_y + 165]

        # stack fluences to create 3D volume
        fluence_3D = np.zeros((165,dose_kernel.shape[1], 165))
        for y in range(dose_kernel.shape[1]):
            fluence_3D[:,y,:] = fl_arr_rotated

        new_aff = np.eye(4)
        new_aff[0,0] = fl_aff_rotated[0,0]
        new_aff[1,1] = slice_thickness
        new_aff[2,2] = fl_aff_rotated[1,1]
        new_aff[0,3] = fl_aff_rotated[0,3]
        new_aff[1,3] = fl_aff_rotated[2,3]
        new_aff[2,3] = fl_aff_rotated[1,3]

        # take attenuation and beam divergence into account
        fluence_3D = fftconvolve(fluence_3D, dose_kernel, mode='same')
    
        # save_path = os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient)
        # if not os.path.exists(save_path):
            # os.makedirs(save_path)
        # nib.save(nib.Nifti1Image(fluence_3D, new_aff), os.path.join(save_path, 'FL' + name_beams[i] + '_volume.nii.gz'))

        # rotate according to gantry angle
        FluenceRotator3D = ImageRotator(fluence_3D, new_aff, - gantry_angles[i], isocenterPosition)
        fl_arr_rotated_3D, fl_aff_rotated_3D = FluenceRotator3D.doInterpolation(1)

        # rotation changes dimensions
        if fl_arr_rotated_3D.shape[0] < 165:
            # pad x direction with zeros
            offset_x = int(np.round((165 - fl_arr_rotated_3D.shape[0]) / 2))
            fl_arr_rotated_3D = np.pad(fl_arr_rotated_3D, ((offset_x, 165 - fl_arr_rotated_3D.shape[0] - offset_x), (0,0), (0,0)), 'constant', constant_values = 0)

        if fl_arr_rotated_3D.shape[1] < dose_kernel.shape[1]:
            offset_y = int(np.round((dose_kernel.shape[1] - fl_arr_rotated_3D.shape[1]) / 2))
            fl_arr_rotated_3D = np.pad(fl_arr_rotated_3D, ((0,0), (offset_y, dose_kernel.shape[1] - fl_arr_rotated_3D.shape[1] - offset_y), (0,0)), 'constant', constant_values = 0)

        if fl_arr_rotated_3D.shape[2] < 165:
            offset_z = int(np.round((165 - fl_arr_rotated_3D.shape[2]) / 2))
            fl_arr_rotated_3D = np.pad(fl_arr_rotated_3D, ((0,0), (0,0), (offset_z, 165 - fl_arr_rotated_3D.shape[2] - offset_z)), 'constant', constant_values = 0)

        offset_x = int(np.round((fl_arr_rotated_3D.shape[0] - 165) / 2))
        offset_y = int(np.round((fl_arr_rotated_3D.shape[1] - dose_kernel.shape[1]) / 2))
        offset_z = int(np.round((fl_arr_rotated_3D.shape[2] - 165) / 2))

        fl_arr_rotated_3D = fl_arr_rotated_3D[offset_x:offset_x + 165, offset_y:offset_y + dose_kernel.shape[1], offset_z:offset_z + 165]
        # nib.save(nib.Nifti1Image(fl_arr_rotated_3D, fl_aff_rotated_3D), os.path.join(save_path, 'FL' + name_beams[i] + '_volume_rotated.nii.gz'))

        fl_volume_arr += fl_arr_rotated_3D

    # change affine matrix of the volume 
    # the center of the image should coincide with the isocenter of the plan  --- vragen aan Wouter
    rtplan = pydicom.dcmread(os.path.join(exporteclipse_data_path_initial_train, patient, 'RTPLAN.dcm'))
    iso = rtplan.BeamSequence[0].ControlPointSequence[0].IsocenterPosition

    center_x = fl_volume_arr.shape[0] // 2
    center_y = fl_volume_arr.shape[1] // 2
    center_z = fl_volume_arr.shape[2] // 2

    fl_volume_aff = np.eye(4)
    fl_volume_aff[0,0] = fl_aff[0,0]
    fl_volume_aff[1,1] = slice_thickness
    fl_volume_aff[2,2] = fl_aff[1,1]
    fl_volume_aff[0,3] = iso[0] - center_x * fl_volume_aff[0,0]
    fl_volume_aff[1,3] = iso[1] - center_y * fl_volume_aff[1,1]
    fl_volume_aff[2,3] = iso[2] - center_z * fl_volume_aff[2,2]

    nib.save(nib.Nifti1Image(fl_volume_arr, fl_volume_aff), os.path.join(preproc_data_path_initial_train, patient, 'FL_maps_volume.nii.gz'))
    fl_volume = nib.load(os.path.join(preproc_data_path_initial_train, patient, 'FL_maps_volume.nii.gz'))

    # center image around center of mass of target volume (isocenter) and crop to fixed size (PhD Siri Willems)
    iso_x = iso[0] # isocenter of the plan, in world coordinates
    iso_y = iso[1]
    iso_z = iso[2]

    delta_x = new_aff[0,0] # affine matrix to go from world coordinates to voxel coordinates
    delta_y = new_aff[1,1]
    delta_z = new_aff[2,2]

    x_img = new_aff[0,3]
    y_img = new_aff[1,3]
    z_img = new_aff[2,3]

    iso_fluence =  [int(np.round((iso_x - x_img) / delta_x)),
                int(np.round((iso_y - y_img) / delta_y)), 
                int(np.round((iso_z - z_img) / delta_z))] # where does iso lay in fluence volume
    
    if iso_fluence[0] < input_size[0] // 2:
        iso_fluence[0] = input_size[0] // 2
    if iso_fluence[1] < input_size[1] // 2:
        iso_fluence[1] = input_size[1] // 2
    if iso_fluence[2] < input_size[2] // 2:    
        iso_fluence[2] = input_size[2] // 2

    fl_volume = fl_volume.slicer[iso_fluence[0] - input_size[0] // 2 : iso_fluence[0] + input_size[0] // 2,
                            iso_fluence[1] - input_size[1] // 2 : iso_fluence[1] + input_size[1] // 2, 
                            iso_fluence[2] - input_size[2] // 2 : iso_fluence[2] + input_size[2] // 2]

    nib.save(fl_volume, os.path.join(preproc_data_path_initial_train, patient, 'FL_maps_volume_reshaped.nii.gz'))

# %% CREATE DOSE VOLUME
for patient in os.listdir(exporteclipse_data_path_initial_train):
    print(patient)

    dose = os.path.join(exporteclipse_data_path_initial_train, patient, 'RTDOSE.dcm')
    nifti_generator = RTDoseVolume(dose)
    dose_arr = nifti_generator.get_data()
    dose_aff = nifti_generator.affine
    nib.save(nib.Nifti1Image(dose_arr, dose_aff), os.path.join(exporteclipse_data_path_initial_train, patient, 'RTDOSE.nii.gz'))

    dose = nib.load(os.path.join(exporteclipse_data_path_initial_train, patient, 'RTDOSE.nii.gz'))
    dose_aff = dose.affine

    rtplan = pydicom.dcmread(os.path.join(exporteclipse_data_path_initial_train, patient, 'RTPLAN.dcm'))
    iso = rtplan.BeamSequence[0].ControlPointSequence[0].IsocenterPosition # world coordinates?

    # center image around center of mass of target volume (isocenter) and crop to fixed size (PhD Siri Willems)
    iso_x = iso[0] # isocenter of the plan, in world coordinates
    iso_y = iso[1]
    iso_z = iso[2]

    delta_x = dose_aff[0,0] # affine matrix to go from world coordinates to voxel coordinates
    delta_y = dose_aff[1,1]
    delta_z = dose_aff[2,2]

    x_img = dose_aff[0,3]
    y_img = dose_aff[1,3]
    z_img = dose_aff[2,3]

    iso_dose =  [int(np.round((iso_x - x_img) / delta_x)),
                int(np.round((iso_y - y_img) / delta_y)), 
                int(np.round((iso_z - z_img) / delta_z))]

    if iso_dose[0] < input_size[0] // 2:
        iso_dose[0] = input_size[0] // 2
    if iso_dose[1] < input_size[1] // 2:
        iso_dose[1] = input_size[1] // 2
    if iso_dose[2] < input_size[2] // 2:    
        iso_dose[2] = input_size[2] // 2

    dose_reshaped = dose.slicer[iso_dose[0] - input_size[0] // 2 : iso_dose[0] + input_size[0] // 2,
                            iso_dose[1] - input_size[1] // 2 : iso_dose[1] + input_size[1] // 2, 
                            iso_dose[2] - input_size[2] // 2 : iso_dose[2] + input_size[2] // 2]

    nib.save(dose_reshaped, os.path.join(preproc_data_path_initial_train, patient, 'RTDOSE_reshaped.nii.gz'))

    '''
    # processing output of network
    ct = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/export_initial_nonclinicaleclipse/train', patient, 'CT_ED.nii.gz'))

    dose_aff = dose_reshaped.affine
    dose_arr = dose_reshaped.get_fdata()

    ct_aff = ct.affine
    ct_arr = ct.get_fdata()

    reformatter = Reformatter3DTo3D(dose_arr, dose_aff, ct_arr, ct_aff)
    dose_reformat_arr, dose_reformat_aff = reformatter.Interpollate()
    nib.save(nib.Nifti1Image(dose_reformat_arr, dose_reformat_aff), os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient, 'RTDOSE_reformat.nii.gz'))
    '''