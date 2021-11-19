from typing import Dict, List, Tuple
import numpy as np
import napari
from tqdm import tqdm
from itertools import product

try:
    from omero.gateway import BlitzGateway
except:
    print("Could not import BlitzGateway which is")
    print("required to download the sample datasets.")
    print("Please install omero-py:")
    print("https://pypi.org/project/omero-py/")
    exit(-1)


def IDR_fetch_image(image_id: int, progressbar: bool = True) -> np.ndarray:
    """
    Download the image with id image_id from the IDR
    Will fetch all image planes corresponding to separate
    timepoints/channels/z-slices and return a numpy
    array with dimension order (t,z,y,x,c)
    Displaying download progress can be disabled by passing
    False to progressbar.
    """

    conn = BlitzGateway(
        host="ws://idr.openmicroscopy.org/omero-ws",
        username="public",
        passwd="public",
        secure=True,
    )
    conn.connect()
    conn.c.enableKeepAlive(60)

    idr_img = conn.getObject("Image", image_id)
    idr_pixels = idr_img.getPrimaryPixels()

    _ = idr_img
    nt, nz, ny, nx, nc = (
        _.getSizeT(),
        _.getSizeZ(),
        _.getSizeY(),
        _.getSizeX(),
        _.getSizeC(),
    )

    plane_indices = list(product(range(nz), range(nc), range(nt)))
    idr_plane_iterator = idr_pixels.getPlanes(plane_indices)

    if progressbar:
        idr_plane_iterator = tqdm(idr_plane_iterator, total=len(plane_indices))

    _tmp = np.asarray(list(idr_plane_iterator))
    _tmp = _tmp.reshape((nz, nc, nt, ny, nx))
    # the following line reorders the axes (no summing, despite the name)
    return np.einsum("jmikl", _tmp)

samples = (
    {"IDRid": 2864587, "description": "AURKB knockdown", "vol": None},
    {"IDRid": 2862565, "description": "KIF11 knockdown", "vol": None},
    {"IDRid": 2867896, "description": "INCENP knockdown", "vol": None},
    {"IDRid": 1486532, "description": "TMPRSS11A knockdown", "vol": None},
)

print("-------------------------------------------------------")
print("Sample datasets will require ~490 MB download from IDR.")
answer = input("Press Enter to proceed, 'n' to cancel: ")
if answer.lower().startswith('n'):
    print("User cancelled download. Exiting.")
    exit(0)
print("-------------------------------------------------------")
for s in samples:
    print(f"Downloading sample {s['IDRid']}.")
    print(f"Description: {s['description']}")
    s["vol"] = np.squeeze(IDR_fetch_image(s["IDRid"]))

with napari.gui_qt():
    v = napari.Viewer(ndisplay=3)
    scale = (5, 1, 1)  # "stretch" time domain
    for s in samples:
        v.add_image(
            s["vol"], name=s['description'], scale=scale, blending="opaque"
        )

    v.grid.enabled = True  # show the volumes in grid mode
    v.axes.visible = True  # magenta error shows time direction

    # set an oblique view angle onto the kymograph grid
    v.camera.center = (440, 880, 1490)
    v.camera.angles = (-20, 23, -50)
    v.camera.zoom = 0.17