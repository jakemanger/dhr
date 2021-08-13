"""
Load a Volume (tif stack).
Invoke a tool to cut off parts of it.
"""
print(__doc__)

from vedo import Plotter, datadir

vp = Plotter()
# Load a mesh and show it
# filename = 'P_crassipes_FEG191022_077A/new/'
filename = datadir+"embryo.tif"

vol = vp.load(filename)
vol = vol.isosurface().normalize().lw(0).c("violet")
vol = vol.lighting('glossy').phong()
vp.addCutterTool(vol)
vp.show(vol)