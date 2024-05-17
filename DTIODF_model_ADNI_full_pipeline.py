interactive = False

from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.data import default_sphere
from dipy.reconst.dti import color_fa
from dipy.direction import peaks_from_model
from dipy.viz import window, actor, has_fury
from dipy.segment.mask import median_otsu
from dipy.tracking.local_tracking import LocalTracking, pft_tracker
from dipy.tracking.streamline import Streamlines
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import (AnatomicalStoppingCriterion,
                                              StreamlineStatus)
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.viz import colormap
import os
import numpy as np
import nibabel as nib


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

# Common bval and bvec files
common_bval_file = np.load('common_bval_file.npy')
common_bvec_file = np.load('common_bvec_file.npy')


# Output folder
output_folder = '41diff_tractography_tck'

# Loop over each subject
for subject_name in subject_folder:
    # Load data
    nii_gz_file = f'ordered_4d_image_{subject_name}.nii.gz'
    data, affine, hardi_img = load_nifti(nii_gz_file, return_img=True)

    # Load bvals and bvecs
    gtab = gradient_table(common_bval_file, common_bvec_file)

    # Print data shape
    print(data.shape)

    # Masking
    maskdata, mask = median_otsu(data, vol_idx=range(10, 46), median_radius=3,
                                 numpass=1, autocrop=False, dilate=2)
    print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

    # Tensor model fitting
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    # Visualization setup
    from dipy.data import get_sphere
    sphere = get_sphere('repulsion724')

    FA = fractional_anisotropy(tenfit.evals)
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, tenfit.evecs)

    evals = tenfit.evals[:, :, 28:29, :]
    evecs = tenfit.evecs[:, :, 28:29, :]


    cfa = RGB[:, :, 28:29, :]
    cfa /= cfa.max()

    # Create scene
    scene = window.Scene()

    # Visualization of tensor ellipsoids
    scene.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere,
                                  scale=0.3))

    # Save illustration as tensor_ellipsoids.png
    illustration_filename = f'tensor_ellipsoids_{subject_name}.png'
    print(f'Saving illustration as {illustration_filename}')
    window.record(scene, n_frames=1, out_path=illustration_filename, size=(600, 600))

    # ODF fitting and visualization
    tensor_odfs = tenmodel.fit(data[:, :, 38:39, :]).odf(sphere)
    odf_actor = actor.odf_slicer(tensor_odfs, sphere=sphere, scale=0.5,
                                 colormap=None)
    scene.add(odf_actor)

    odf_filename = f'tensor_odfs_{subject_name}.png'
    print(f'Saving illustration as {odf_filename}')
    window.record(scene, n_frames=1, out_path=odf_filename, size=(600, 600))

    # Peaks from model
    maskdata1 = maskdata[:, :, :, 0]
    dti_peaks = peaks_from_model(tenmodel, data, sphere=sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45, normalize_peaks=False,
                                 mask=maskdata1, sh_order=8, sh_basis_type=None, npeaks=1)

    # Visualize peaks
    if has_fury:
        scene = window.Scene()
        scene.add(actor.peak_slicer(dti_peaks.peak_dirs,
                                    dti_peaks.peak_values,
                                    colors=None))
        peaks_filename = f'peaks_{subject_name}.png'
        print(f'Saving illustration as {peaks_filename}')
        window.record(scene, n_frames=1, out_path=peaks_filename, size=(600, 600))

    # Tractography
    FA_mask = FA > 0.2
    FA_mask1 = FA > 0.8
    seed_mask = FA_mask1

    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

    stopping_criterion = ThresholdStoppingCriterion(FA, .25)
    seeds = utils.seeds_from_mask(seed_mask, density=2, affine=affine)

    streamlines_generator = LocalTracking(dti_peaks, stopping_criterion, seeds,
                                          affine=affine, step_size=.5)

    streamlines = Streamlines(streamlines_generator)

    # Visualize streamlines
    if has_fury:
        scene = window.Scene()
        streamlines_actor = actor.line(streamlines,
                                       colormap.line_colors(streamlines))
        streamlines_filename = f'streamlines_{subject_name}.png'
        scene.add(streamlines_actor)

        print(f'Saving illustration as {streamlines_filename}')
        window.record(scene, n_frames=1, out_path=streamlines_filename, size=(800, 800))

    # Save TCK file
    sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
    filename = f'tractogram_deterministic_{subject_name}_DTI_ADNI.trk'
    output_filepath = os.path.join(output_folder, filename)
    save_trk(sft, output_filepath)

    import numpy as np
    from dipy.tracking.streamline import transform_streamlines
    from dipy.core.interpolation import interpolate_scalar_3d

    # Save coordinates
    transformed_streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))
    coordinates = np.concatenate([s.tolist() for s in transformed_streamlines]).astype(float)
    coordinates = coordinates.reshape(-1, 3)

    interpolated_values, inside = interpolate_scalar_3d(FA, coordinates)
    valid_coordinates = coordinates[inside == 1]

    output_folder1 = ''
    # Save coordinates to a text file
    filename_coordinates = f'coordinates_{subject_name}.txt'
    output_filepath_coordinates = os.path.join(output_folder1, filename_coordinates)
    np.savetxt(output_filepath_coordinates, valid_coordinates, delimiter=',')
