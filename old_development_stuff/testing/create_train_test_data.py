import numpy as np
import napari
import matplotlib
from data_preparation import load_data, get_distance_prob_to_point, split_data_into_windows

# load datasets
mct, corneas, rhabdoms = load_data(
    'data/P_crassipes_FEG191022_077A/new/*.dcm',
    'P_crassipes_FEG191022_077A.mat'
)
cornea_prob, rhabdom_prob = get_distance_prob_to_point(
    mct,
    corneas,
    rhabdoms
)
del corneas, rhabdoms

# plot data
with napari.gui_qt():
    print('plotting loaded data...')
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3
    viewer.add_image(mct, contrast_limits=(18000, 32800))
    # viewer.add_points(corneas, name='Corneas')
    # viewer.add_points(rhabdoms, name='Rhabdoms')
    viewer.add_image(
        cornea_prob, colormap='cyan', opacity=0.5, contrast_limits=(0.01, 1))
    viewer.add_image(
        rhabdom_prob, colormap='magma', opacity=0.5, contrast_limits=(0.01, 1))

# TODO normalise values in mct images


# TODO repeat this multiple times by rotating the stack
# iteratively in a for loop
# TODO add a buffer to allow edges of data to be analysed

mct = mct.compute()

window_size = 31
step_size = 4
centroid_vox = (window_size - 1) / 2
assert isinstance(centroid_vox, int), (
    'window_size should be an odd integer, so centroid_vox is an integer')

# now split data into smaller windows


# X variable
window_mct = split_data_into_windows(mct)
del mct
# Y variable
window_cornea_prob = split_data_into_windows(cornea_prob)
del cornea_prob
window_rhabdom_prob = split_data_into_windows(rhabdom_prob)
del rhabdom_prob
validation = np.column_stack(
    (
        window_cornea_prob[:, centroid_vox, centroid_vox, centroid_vox],
        window_rhabdom_prob[:, centroid_vox, centroid_vox, centroid_vox]
    )
)

# lets plot some example rhabdoms and corneas
# to make sure everything worked
print('making final grid plot of some cornea and rhabdom windows...')
with napari.gui_qt():
    # plot with multiple 3d images of rhabdoms and corneas
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3 # toggle 3 dimensional view
    cornea_indx = np.where(validation[:, 0] > 0.95)[0]
    rhabdom_indx = np.where(validation[:, 1] > 0.95)[0]
    for i in range(0, 10): 
        viewer.add_image(
            window_mct[cornea_indx[i], :, :, :],
            contrast_limits=(18000, 32800),
            name='cornea window ' + i
        )
        viewer.add_image(
            window_mct[rhabdom_indx[i], :, :, :],
            contrast_limits=(18000, 32800),
            name='rhabdom window ' + i
        )
    viewer.grid_view()
