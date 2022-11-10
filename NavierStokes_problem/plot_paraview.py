# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
upvd = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
upvdDisplay = Show(upvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
upvdDisplay.Representation = 'Surface'
upvdDisplay.ColorArrayName = [None, '']
upvdDisplay.SelectTCoordArray = 'None'
upvdDisplay.SelectNormalArray = 'None'
upvdDisplay.SelectTangentArray = 'None'
upvdDisplay.OSPRayScaleArray = 'f_80-0'
upvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
upvdDisplay.SelectOrientationVectors = 'f_80-0'
upvdDisplay.ScaleFactor = 0.6283185307179586
upvdDisplay.SelectScaleArray = 'None'
upvdDisplay.GlyphType = 'Arrow'
upvdDisplay.GlyphTableIndexArray = 'None'
upvdDisplay.GaussianRadius = 0.031415926535897934
upvdDisplay.SetScaleArray = ['POINTS', 'f_80-0']
upvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
upvdDisplay.OpacityArray = ['POINTS', 'f_80-0']
upvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
upvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
upvdDisplay.PolarAxes = 'PolarAxesRepresentation'
upvdDisplay.ScalarOpacityUnitDistance = 0.26111849556003236
upvdDisplay.OpacityArrayName = ['POINTS', 'f_80-0']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
upvdDisplay.ScaleTransferFunction.Points = [-99.86536850398117, 0.0, 0.5, 0.0, 3827.9663351593395, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
upvdDisplay.OpacityTransferFunction.Points = [-99.86536850398117, 0.0, 0.5, 0.0, 3827.9663351593395, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=upvd)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [3.141592653589793, 1.0, 1.0471975511965974]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [3.141592653589793, 1.0, 1.0471975511965974]

# show data in view
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = [None, '']
slice1Display.SelectTCoordArray = 'None'
slice1Display.SelectNormalArray = 'None'
slice1Display.SelectTangentArray = 'None'
slice1Display.OSPRayScaleArray = 'f_80-0'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'f_80-0'
slice1Display.ScaleFactor = 0.2094395102393195
slice1Display.SelectScaleArray = 'None'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'None'
slice1Display.GaussianRadius = 0.010471975511965975
slice1Display.SetScaleArray = ['POINTS', 'f_80-0']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'f_80-0']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [-62.63335933786804, 0.0, 0.5, 0.0, 3816.145313543876, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [-62.63335933786804, 0.0, 0.5, 0.0, 3816.145313543876, 1.0, 0.5, 0.0]

# hide data in view
Hide(upvd, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Warp By Vector'
warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=slice1)
warpByVector1.Vectors = ['POINTS', 'f_80-0']

# show data in view
warpByVector1Display = Show(warpByVector1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
warpByVector1Display.Representation = 'Surface'
warpByVector1Display.ColorArrayName = [None, '']
warpByVector1Display.SelectTCoordArray = 'None'
warpByVector1Display.SelectNormalArray = 'None'
warpByVector1Display.SelectTangentArray = 'None'
warpByVector1Display.OSPRayScaleArray = 'f_80-0'
warpByVector1Display.OSPRayScaleFunction = 'PiecewiseFunction'
warpByVector1Display.SelectOrientationVectors = 'f_80-0'
warpByVector1Display.ScaleFactor = 387.87786728817446
warpByVector1Display.SelectScaleArray = 'None'
warpByVector1Display.GlyphType = 'Arrow'
warpByVector1Display.GlyphTableIndexArray = 'None'
warpByVector1Display.GaussianRadius = 19.393893364408722
warpByVector1Display.SetScaleArray = ['POINTS', 'f_80-0']
warpByVector1Display.ScaleTransferFunction = 'PiecewiseFunction'
warpByVector1Display.OpacityArray = ['POINTS', 'f_80-0']
warpByVector1Display.OpacityTransferFunction = 'PiecewiseFunction'
warpByVector1Display.DataAxesGrid = 'GridAxesRepresentation'
warpByVector1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
warpByVector1Display.ScaleTransferFunction.Points = [-62.63335933786804, 0.0, 0.5, 0.0, 3816.145313543876, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
warpByVector1Display.OpacityTransferFunction.Points = [-62.63335933786804, 0.0, 0.5, 0.0, 3816.145313543876, 1.0, 0.5, 0.0]

# hide data in view
Hide(slice1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on warpByVector1
warpByVector1.ScaleFactor = 0.00020817074795014147

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(warpByVector1Display, ('POINTS', 'f_80-0', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
warpByVector1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
warpByVector1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'f_800'
f_800LUT = GetColorTransferFunction('f_800')

# get opacity transfer function/opacity map for 'f_800'
f_800PWF = GetOpacityTransferFunction('f_800')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1156, 677)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [10.507026496279268, -9.830251501645458, 3.7099605809158116]
renderView1.CameraFocalPoint = [3.141592653589794, 1.0, 1.0471975511965967]
renderView1.CameraViewUp = [-0.0005101099373264195, 0.2384259686225679, 0.9711605414524636]
renderView1.CameraParallelScale = 3.45922348400931

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).