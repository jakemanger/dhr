"""
Click anywhere on the mesh and press c.
"""
from vedo import *
import vtk
from vedo.colors import printc

camera_focal_point = [0, 0, 0]

def myfnc(key):
    if key=='a':
       camera_focal_point[1] -= 0.001
       vp.camera.SetFocalPoint(camera_focal_point) 

    if key=='d':
       camera_focal_point[1] += 0.001
       vp.camera.SetFocalPoint(camera_focal_point) 

    mesh = vp.clickedActor
    if not mesh or key != "c":
        printc("click a mesh and press c.", c="r")
        return
    printc("clicked mesh    :", mesh.filename[-40:], c=4)
    printc("clicked 3D point:", mesh.picked3d, c=4)
    printc("clicked renderer:", [vp.renderer], c=2)

    # vp.camera.SetPosition( [6.316, -3.586, 1.36] )
    # vp.camera.SetFocalPoint( [-0.195, -0.762, -0.802] )
    # vp.camera.SetViewUp( [-0.245, 0.166, 0.955] )
    # vp.camera.SetDistance( 7.42 )
    # vp.camera.SetClippingRange( [4.283, 11.386] )

    vp.add(Sphere(pos=mesh.picked3d, r=0.004, c="y"))


vp = Plotter()

vp.keyPressFunction = myfnc  # make it known to Plotter class

def _turn_to_mesh(vol):
    print('converting volume to mesh...')
    vol = vol.isosurface().normalize().lw(0).c("violet")
    vol = vol.clone().triangulate().smoothLaplacian()
    # mesh_lighting = input('enter mesh lighting style. Can be `[metallic, plastic, shiny, glossy]`\n')
    mesh_lighting = 'metallic' 
    vol = vol.lighting(mesh_lighting).phong()
    print('mesh created')
    return(vol)

vol = vp.load(datadir+"bunny.obj")
# vol = vp.load('P_crassipes_FEG191022_077A/new/')
# vol = _turn_to_mesh(vol)

printc("\nPress c to execute myfnc()", c=1)
vp.show(vol, interactive=True, axes=1, interactorStyle=3)