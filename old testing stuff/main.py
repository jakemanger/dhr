from vedo import *
import vedo.applications as applications

vp = Plotter()
# Load a mesh and show it
filename = 'P_crassipes_FEG191022_077A/new/'
vol = vp.load(filename)


sp = vol.spacing()
vol.spacing([sp[0]*1, sp[1]*1, sp[2]*1])
vol.mode(int(0)).color("jet").jittering(True)
vp = applications.RayCaster(vol)
vp.show(viewup="z", interactive=True)
vp.sliders[0][0].SetEnabled(False)
vp.sliders[1][0].SetEnabled(False)
vp.sliders[2][0].SetEnabled(False)



# # turn to mesh
# vol = vol.isosurface().normalize().lw(0).c("violet")
# vol = vol.clone().triangulate().smoothLaplacian()
# vol = vol.lighting('glossy').phong()

# # # add a button to the current renderer (e.i. nr1)
# # def buttonfunc():
# #     vp.addCutterTool(vol)

# #     bu.switch()                 # change to next status
# #     printc(bu.status(), box="_", dim=True)


# # bu = vp.addButton(
# #     buttonfunc,
# #     pos=(0.7, 0.05),  # x,y fraction from bottom left corner
# #     states=["press to enable cutter tool", "press to disable cutter tool"],
# #     c=["w", "w"],
# #     bc=["dg", "dv"],  # colors of states
# #     font="courier",   # arial, courier, times
# #     size=25,
# #     bold=True,
# #     italic=False,
# # )

# vp.addCutterTool(vol)

# vp.show()