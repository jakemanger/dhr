import sys
from PyQt5 import Qt

# You may need to uncomment these lines on some systems:
#import vtk.qt
#vtk.qt.QVTKRWIBase = "QGLWidget"

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vedo import Plotter, Sphere

from vedo.colors import printc

class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        # create renderer and add the actors
        self.vp = Plotter(qtWidget=self.vtkWidget, interactive=True)
        vol = self._load_image('P_crassipes_FEG191022_077A/new/')
        vol = self._turn_to_mesh(vol)
        self.vp.keyPressFunction = self._myfnc  # make myfnc known to Plotter class
        self.vp.show(vol)

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.iren.AddObserver("LeftButtonPressEvent", self.vp._mouseleft)
        self.iren.AddObserver("RightButtonPressEvent", self.vp._mouseright)
        self.iren.AddObserver("MiddleButtonPressEvent", self.vp._mousemiddle)


        # set-up the rest of the Qt window
        button = Qt.QPushButton("Turn red")
        button.setToolTip('This is an example button')
        button.clicked.connect(self.onClick)
        self.vl.addWidget(button)
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show() # <--- show the Qt Window

    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        self.vtkWidget.close()

    @Qt.pyqtSlot()
    def onClick(self):
        self.vp.actors[0].color('red')
        self.vp.interactor.Render()

    def _load_image(self, filename):
        vol = self.vp.load(filename)
        return(vol)

    def _turn_to_mesh(self, vol):
        print('converting volume to mesh...')
        vol = vol.isosurface().normalize().lw(0).c("violet")
        vol = vol.clone().triangulate().smoothLaplacian()
        vol = vol.lighting('glossy').phong()
        print('mesh created')
        return(vol)
        
    def _myfnc(self, key):
        mesh = self.vp.clickedActor
        if not mesh or key != "c":
            printc("click a mesh and press c.", c="r")
            return
        printc("clicked mesh    :", mesh.filename[-40:], c=4)
        printc("clicked 3D point:", mesh.picked3d, c=4)
        printc("clicked renderer:", [self.vp.renderer], c=2)

        self.vp.add(Sphere(pos=mesh.picked3d, r=0.004, c="v"))


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()