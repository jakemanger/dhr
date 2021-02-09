from dask_image.imread import imread
import napari

stack = imread("P_crassipes_FEG191022_077A/new/*.dcm")
stack

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(stack)