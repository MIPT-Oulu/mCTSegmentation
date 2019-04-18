import vtk
from vtk.util import numpy_support
import numpy as np


def numpy2vtk(stack, spacing, color):
    dims = stack.shape
    stack64bit = stack.ravel().astype(np.float64) * color
    vtk_array = numpy_support.numpy_to_vtk(stack64bit, array_type=vtk.VTK_FLOAT)
    vtk_data = vtk.vtkImageData()
    vtk_data.SetExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
    # Use spacing != 1, if axes dimensions should differ from data dimensions
    vtk_data.SetSpacing(spacing, spacing, spacing)
    # Set vtk array to image data
    vtk_data.GetPointData().SetScalars(vtk_array)
    return vtk_data


def visualize_stack(stack: np.ndarray, color=1., spacing=1., bg='white', wsize=None, cam_pos=None, cam_fp=None):
    if bg == 'white':
        bg = (1, 1, 1)
    elif bg == 'black':
        bg = (0, 0, 0)
    elif not isinstance(bg, tuple):
        raise NotImplementedError('BG color should be white, black or tuple')

    if wsize is None:
        wsize = (800, 800)
    if cam_pos is None:
        cam_pos = (2.0, -4.0, 4.0)

    if cam_fp is None:
        cam_fp = (0.5, 0.5, 0.5)

    vol = vtk.vtkVolume()
    mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    ctf = vtk.vtkColorTransferFunction()
    scalar_opacity = vtk.vtkPiecewiseFunction()
    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    iren = vtk.vtkRenderWindowInteractor()

    dims = stack.shape
    vtkdata = numpy2vtk(stack, color, spacing)

    mapper.SetInputData(vtkdata)
    mapper.SetBlendModeToComposite()
    mapper.Update()

    # Connect mapper to volume actor
    vol.SetMapper(mapper)

    # Set color from a linear color transfer function
    ctf.AddRGBPoint(0, 0, 0, 0)
    ctf.AddRGBPoint(1 * color, 0.9 * color, 0.9 * color, 0.9 * color)

    # Set opacity from a piecewise function
    scalar_opacity.AddPoint(0, 0)
    scalar_opacity.AddPoint(0.1 * color, 0)
    scalar_opacity.AddPoint(0.25 * color, 0.1)
    scalar_opacity.AddPoint(0.5 * color, 0.8)
    scalar_opacity.AddPoint(1 * color, 1)

    vol.GetProperty().SetColor(ctf)
    vol.GetProperty().SetScalarOpacity(scalar_opacity)
    vol.GetProperty().SetInterpolationTypeToLinear()
    vol.GetProperty().ShadeOn()
    vol.GetProperty().SetAmbient(0.6)
    vol.GetProperty().SetDiffuse(0.6)
    vol.GetProperty().SetSpecular(0.2)

    # Update volume actor
    vol.Update()

    # Connect volume to renderer and set background
    renderer.SetBackground(bg[0], bg[1], bg[2])
    renderer.AddVolume(vol)

    renderer.GetActiveCamera().Azimuth(90)
    renderer.GetActiveCamera().SetViewUp(0, 0, 1)

    renderer.GetActiveCamera().SetPosition(dims[2] * cam_pos[2] * spacing,
                                           dims[1] * cam_pos[1] * spacing,
                                           dims[0] * cam_pos[0] * spacing)

    renderer.GetActiveCamera().SetFocalPoint(dims[2] * cam_fp[2] * spacing,
                                             dims[1] * cam_fp[1] * spacing,
                                             dims[0] * cam_fp[0] * spacing)

    # Connect renderer to render window
    renWin.AddRenderer(renderer)
    renWin.SetSize(wsize[0], wsize[1])

    # Connect window to interactor
    iren.SetRenderWindow(renWin)

    # Set interactor style
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # Render
    renWin.Render()
    renWin.GetRenderers().GetFirstRenderer().ResetCamera()

    iren.Start()