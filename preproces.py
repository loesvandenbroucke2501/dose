import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import scipy.ndimage
from scipy.signal import fftconvolve
import math
from scipy.interpolate import RegularGridInterpolator

import sys
sys.path.append('/uz/data/radiotherapie/Shared/Data/Liesbeth_Vdw/phd_projects/general/dicomTONifty')
from DICOMtoNIFTI.Read_RTDose import RTDoseVolume
from DICOMtoNIFTI.reformat import Reformatter3DTo3D
from DICOMtoNIFTI.Read_Fluencetxt import FluenceHeader
from DICOMtoNIFTI.preprocessing_gantryrotation import ImageRotator2D

fluence = nib.load('/uz/data/radiotherapie/Shared/Data/LoesVdb/data_preproc_initial/train/0057_DC/FL_maps_rotated.nii.gz')
fl_arr = fluence.get_fdata()[:,:,0]
fl_aff = fluence.affine
angle = 90

header = FluenceHeader('/uz/data/radiotherapie/Shared/Data/LoesVdb/fluenceheaders/data_preproc_initial/train/0057_DC/FL_LAO30_header.txt')
isocenter_position = header.iso

FluenceRotator = ImageRotator2D(fl_arr, fl_aff, angle, isocenter_position)
fl_arr_rotated, fl_aff_rotated = FluenceRotator.doInterpolation(1)

nib.save(nib.Nifti1Image(fl_arr_rotated, fl_aff_rotated), '/uz/data/radiotherapie/Shared/Data/LoesVdb/data_preproc_initial/train/0057_DC/FL_TEST.nii.gz')


fixed_size = [160, 160, 160]

# dose kernel
'''
dose = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/F_phantom_fluence_profile_lvdnbrz_test/'
dose_path = os.path.join(dose, 'RTDOSE.dcm')
nifti_generator = RTDoseVolume(dose_path)
dose_arr = nifti_generator.get_data()
dose_aff = nifti_generator.affine
nifti_generator.SaveAsNifti(dose, dose_arr, dose_aff)

field_size = 5 # mm
pixdim = 2.5 # mm
center_x = dose_arr.shape[0] // 2
center_z = dose_arr.shape[2] // 2
nb_pixels = int(field_size / pixdim) + 10 # 10 is margin for beam divergence

dose_kernel = dose_arr[center_x - nb_pixels//2:center_x + nb_pixels//2, :, center_z - nb_pixels//2:center_z + nb_pixels//2]
nib.save(nib.Nifti1Image(dose_kernel, dose_aff), os.path.join(dose, 'dose_kernel.nii.gz'))
'''

dose = nib.load('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/F_phantom_fluence_profile_lvdnbrz_test/dose_kernel.nii.gz')
dose_kernel = dose.get_fdata()
dose_aff = dose.affine

eclipse_data_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data'
train = 'train'
validatie = 'validatie'
test = 'test'
initial = 'data_preproc_initial'
adaptive = 'data_preproc_adaptive'

eclipse_data_train_initial = os.path.join(eclipse_data_path, train)
fluenceheaders = '/uz/data/radiotherapie/Shared/Data/LoesVdb/fluenceheaders/data_preproc_initial/train'


'''
fluenceheaders_data_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/train'
fluenceheaders_save_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/fluenceheaders/data_preproc_initial/train'
if not os.path.exists(fluenceheaders_save_path):
    os.makedirs(fluenceheaders_save_path)

import shutil
for path in os.listdir(fluenceheaders_data_path):

    patient = path.replace('_FluencePred2_Lung_R2','')
    print(patient)

    save_path = os.path.join(fluenceheaders_save_path, patient)
    if not os.path.exists(save_path):
        os.makedirs(save_path)  

    for file in os.listdir(os.path.join(fluenceheaders_data_path, path)):
        if file.endswith('_header.txt'):
            shutil.move(os.path.join(fluenceheaders_data_path, path, file), os.path.join(save_path))

'''


# create fluence volume
for path in os.listdir(eclipse_data_train_initial):

    patient = path.replace('_FluencePred2_Lung_R2','')
    print(patient)

    fm = os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/data_preproc_initial/train', patient, 'FL_maps_rotated.nii.gz')
    fm_arr = nib.load(fm).get_fdata()
    fm_aff = nib.load(fm).affine

    name_beams = ['LAO30', 'RAO345', 'RAO320', 'RAO295', 'Re270', 'RPO245', 'RPO220', 'RPO195', 'LPO150'] # logische volgorde qua plaatsing rond patient
    gantry_angles = [30, 345, 320, 295, 270, 245, 220, 195, 150]

    fluence_volume = np.zeros((165,dose_kernel.shape[1],165))
    slice_thickness = dose_aff[1,1]

    for i in range(fm_arr.shape[2]):

        fluence_2D = fm_arr[:,:,i]

        # fluence map roteren volgens collimator angle
        # in preprocessing is de fluence map geroteerd volgens collimatorangle om te passen in het coordinatensysteem van de BEV
        # nu terug roteren om te passen in het coordinatensysteem van de CT (3D volume)

        
        header = open(os.path.join(fluenceheaders, patient, 'FL_' + name_beams[i] + '_header.txt'), 'r')
        for header_line in header.readlines():
            if header_line.startswith('CollimatorAngle'):
                collimatorAngle = float(header_line.split("\t")[1])

        FluenceRotator = ImageRotator2D(fluence_2D, fm_aff, - collimatorAngle, isocenter_position)
        fl_arr_rotated, fl_aff_rotated = FluenceRotator.doInterpolation(1)
        

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

        # take attenuation into account
        # fluence_3D = scipy.ndimage.convolve(fluence_3D, dose_kernel)  
        fluence_3D = fftconvolve(fluence_3D, dose_kernel, mode='same')
        
        save_path = os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        nib.save(nib.Nifti1Image(fluence_3D, new_aff), os.path.join(save_path, 'FL' + name_beams[i] + '_volume.nii.gz'))

        # rotate according to gantry angle
        # TO DO: ROTATE GAAT OOK AFFIENE MATRIX VERANDEREN, dus klopt nog niet helemaal

        rotated_fluence_3D_arr = scipy.ndimage.rotate(fluence_3D, angle = gantry_angles[i], axes = (0,1), reshape = False)
        nib.save(nib.Nifti1Image(rotated_fluence_3D_arr, new_aff), os.path.join(save_path, 'FL' + name_beams[i] + '_volume_rotated.nii.gz'))

        fluence_volume += rotated_fluence_3D_arr

    # nog resamplen zodanig dat voxel size hetzelfde is van alle images
    from  DICOMtoNIFTI.preprocessing import ImageResampling
    Resampler = ImageResampling(fluence_volume)
    voxel_spacing = [new_aff[0,0], new_aff[1,1], new_aff[2,2]]
    print(voxel_spacing)
    target_spacing = [2.5, 2.5, 2.5]
    fluence_volume_resampled, fluence_volume_resampled_aff = Resampler.transform(voxel_spacing, target_spacing, new_aff, 1)

    nib.save(nib.Nifti1Image(fluence_volume_resampled, fluence_volume_resampled_aff), os.path.join(save_path, 'FL_maps_volume.nii.gz'))

    fluence_volume = nib.load(os.path.join(save_path, 'FL_maps_volume.nii.gz'))

    # reshape fluence volume so that all inputs have the same size
    rtplan_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/export_initial_nonclinicaleclipse/train' 
    dcmheader = pydicom.dcmread(os.path.join(rtplan_path, patient, 'RTPLAN.dcm'))
    iso = dcmheader.BeamSequence[0].ControlPointSequence[0].IsocenterPosition # world coordinates?

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

    if iso_fluence[0] < fixed_size[0] // 2:
        # print(' x van patient', patient)
        iso_fluence[0] = fixed_size[0] // 2
    if iso_fluence[1] < fixed_size[1] // 2:
        # print(' y van patient', patient)
        iso_fluence[1] = fixed_size[1] // 2
    if iso_fluence[2] < fixed_size[2] // 2:    
        # print(' z van patient', patient)
        iso_fluence[2] = fixed_size[2] // 2

    fluence_volume_reshaped = fluence_volume.slicer[iso_fluence[0] - fixed_size[0] // 2 : iso_fluence[0] + fixed_size[0] // 2,
                            iso_fluence[1] - fixed_size[1] // 2 : iso_fluence[1] + fixed_size[1] // 2, 
                            iso_fluence[2] - fixed_size[2] // 2 : iso_fluence[2] + fixed_size[2] // 2]

    nib.save(fluence_volume_reshaped, os.path.join(save_path, 'FL_maps_volume_reshaped.nii.gz'))


    # ct = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/export_initial_nonclinicaleclipse/train', patient, 'CT_ED.nii.gz'))
    # ct_arr = ct.get_fdata()
    # ct_aff = ct.affine

    # fluence eens reformatten, zodat in dezelfde dimensies als de ct
    # reformatter = Reformatter3DTo3D(fluence_volume_reshaped.get_fdata(), fluence_volume_reshaped.affine, ct_arr, ct_aff)
    # fluence_volume_reformat_arr, fluence_volume_reformat_aff = reformatter.Interpollate()
    # nib.save(nib.Nifti1Image(fluence_volume_reformat_arr, fluence_volume_reformat_aff), os.path.join(save_path, 'FL_maps_volume_reformat.nii.gz'))


# create dose volume (.dcm to .nii.gz)
# preprocess dose
# center image around the center of mass of the target volume & crop it to a fixed size

for path in os.listdir(eclipse_data_train_initial):

    patient = path.replace('_FluencePred2_Lung_R2','')
    print(patient)

    # dose = os.path.join(eclipse_data_train_initial, patient + '_FluencePred2_Lung_R2', 'RTDOSE.dcm')
    # nifti_generator = RTDoseVolume(dose)
    # dose_arr = nifti_generator.get_data()
    # dose_aff = nifti_generator.affine


    # tijdelijk save path, ik moet nog een betere manier vinden om de data te organiseren
    # nifti_generator.SaveAsNifti(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient), dose_arr, dose_aff)

    dose = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient, 'RTDOSE.nii.gz'))
    # print(dose.shape)

    rtplan_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/export_initial_nonclinicaleclipse/train' 
    dcmheader = pydicom.dcmread(os.path.join(rtplan_path, patient, 'RTPLAN.dcm'))
    iso = dcmheader.BeamSequence[0].ControlPointSequence[0].IsocenterPosition # world coordinates?

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
    # print(iso_dose)

    # center image around iso (center of mass of target volume) and crop to fixed size

    if iso_dose[0] < fixed_size[0] // 2:
        # print(' x van patient', patient)
        iso_dose[0] = fixed_size[0] // 2
    if iso_dose[1] < fixed_size[1] // 2:
        # print(' y van patient', patient)
        iso_dose[1] = fixed_size[1] // 2
    if iso_dose[2] < fixed_size[2] // 2:    
        # print(' z van patient', patient)
        iso_dose[2] = fixed_size[2] // 2

    dose_reshaped = dose.slicer[iso_dose[0] - fixed_size[0] // 2 : iso_dose[0] + fixed_size[0] // 2,
                            iso_dose[1] - fixed_size[1] // 2 : iso_dose[1] + fixed_size[1] // 2, 
                            iso_dose[2] - fixed_size[2] // 2 : iso_dose[2] + fixed_size[2] // 2]
    # print(dose_reshaped.shape)

    # nib.save(nib.Nifti1Image(dose_reshaped, dose_aff), os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient, 'RTDOSE_reshaped.nii.gz'))
    nib.save(dose_reshaped, os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient, 'RTDOSE_reshaped.nii.gz'))

    ct = nib.load(os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/export_initial_nonclinicaleclipse/train', patient, 'CT_ED.nii.gz'))

    dose_aff = dose_reshaped.affine
    dose_arr = dose_reshaped.get_fdata()

    ct_aff = ct.affine
    ct_arr = ct.get_fdata()

    reformatter = Reformatter3DTo3D(dose_arr, dose_aff, ct_arr, ct_aff)
    dose_reformat_arr, dose_reformat_aff = reformatter.Interpollate()
    nib.save(nib.Nifti1Image(dose_reformat_arr, dose_reformat_aff), os.path.join('/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data', patient, 'RTDOSE_reformat.nii.gz'))