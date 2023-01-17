import warnings
try:
    from fenicstools.Interpolation import interpolate_nonmatching_mesh, interpolate_nonmatching_mesh_any
except:
    warnings.warn("interpolate_nonmatching_mesh/interpolate_nonmatching_mesh_any not installed")

