import os
import ants
import shutil
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti

subject_folder = ["003_S_4441_CN_F_69","003_S_4350_CN_M_73","109_S_4499_CN_M_84",
                  "003_S_4288_CN_F_73","003_S_4644_CN_F_68","007_S_4620_CN_M_77",
                  "098_S_4018_CN_M_76","098_S_4003_CN_F_78","094_S_4234_CN_M_70",
                  "021_S_4335_CN_F_73","021_S_4276_CN_F_75","016_S_2284_EMCI_M_73",
                  "016_S_4575_EMCI_F_62","016_S_2007_EMCI_F_84","005_S_4185_EMCI_M_81",
                  "007_S_2106_EMCI_M_81","005_S_2390_EMCI_F_89","016_S_4902_LMCI_F_77",
                  "016_S_4646_LMCI_F_62","016_S_4584_LMCI_F_78","007_S_4611_LMCI_M_68",
                  "003_S_4524_LMCI_M_72","003_S_4354_LMCI_M_76","021_S_4857_LMCI_M_68",
                  "003_S_2374_F_82_EMCI","057_S_4909_F_78_LMCI","003_S_4119_M_79_CN",
                  "057_S_4897_F_76_EMCI","094_S_2367_M_75_EMCI","003_S_4081_F_73_CN",
                  "094_S_2216_M_69_EMCI","094_S_4162_F_72_LMCI","094_S_4503_F_72_CN",
                  "094_S_4630_F_66_LMCI","098_S_0896_M_86_CN","098_S_2047_M_78_EMCI",
                  "094_S_4858_M_57_EMCI","098_S_2052_M_74_EMCI","094_S_4295_F_70_LMCI",
                  "094_S_4486_F_69_EMCI","094_S_4560_F_70_CN","098_S_2071_M_85_EMCI",
                  "098_S_2079_M_66_EMCI","098_S_4002_F_74_CN","098_S_4059_M_72_EMCI",
                  "027_S_4729_LMCI_F_78","021_S_4633_LMCI_F_73","021_S_4402_LMCI_F_73",
                  "005_S_4168_EMCI_M_82","007_S_4272_EMCI_M_72","016_S_2031_EMCI_M_73",
                  "016_S_4097_CN_F_71","007_S_4516_CN_M_72","007_S_4488_CN_M_73",
                  "005_S_0610_CN_M_89","003_S_4872_CN_F_69","003_S_4840_CN_M_62",
                  "003_S_4839_CN_M_66","003_S_4555_CN_F_66","005_S_4707_M_68_AD",
                  "005_S_5119_F_77_AD","003_S_4142_F_90_AD","003_S_5165_M_79_AD",
                  "003_S_4152_M_61_AD","005_S_5038_M_82_AD","005_S_4910_F_82_AD",
                  "003_S_4136_M_67_AD","003_S_4892_F_75_AD"]

# Paths that remain constant
moving_image_path = "FMRIB58_FA_1mm.nii.gz"
label_image_path = "JHU-ICBM-labels-1mm.nii.gz"
warped_folder = "registered_data/warped"

for subject_id in subject_folder:
    fixed_image_path = f"AD_data/DTI_parameters_41_Diff_AD/{subject_id}/tensor_fa_{subject_id}.nii.gz"
    data, affine, hardi_img = load_nifti(fixed_image_path, return_img=True)
    output_label_path = f"registered_data/registered_label_image_{subject_id}.nii.gz"

    fixed = ants.image_read(fixed_image_path)
    moving = ants.image_read(moving_image_path)

    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')

    warp_path = mytx['fwdtransforms'][0]
    affine_path = mytx['fwdtransforms'][1]

    # Apply transformations to the label image
    label_image = ants.image_read(label_image_path)
    transformed_label_image = ants.apply_transforms(fixed=fixed, moving=label_image, interpolator='nearestNeighbor', transformlist=mytx['fwdtransforms'])

    # Save the transformed label image
    ants.image_write(transformed_label_image, output_label_path)

    # Save the transformation matrix to a file
    affine_transform_path = mytx['fwdtransforms'][0]
    warp = os.path.join(warped_folder, f"{subject_id}_warp.nii.gz")
    affine_mat = os.path.join(warped_folder, f"{subject_id}_mat.mat")
