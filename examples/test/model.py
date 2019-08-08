from examples.seismic import Model
import numpy as np

def demo_model(preset, **kwargs):
    # A simple circle in a 2D domain with a background velocity.
    # By default, the circle velocity is 2.5 km/s,
    # and the background veloity is 3.0 km/s.
    space_order = kwargs.pop('space_order', 2)
    
    dtype = kwargs.pop('dtype', np.float32)
    shape = kwargs.pop('shape', (101, 101))
    spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))
    nbpml = kwargs.pop('nbpml', 10)
    vp = kwargs.pop('vp', 3.0)
    vp_background = kwargs.pop('vp_background', 2.5)
    r = kwargs.pop('r', 15)

    # Only a 2D preset is available currently
    assert(len(shape) == 2)

    v = np.empty(shape, dtype=dtype)
    v[:] = vp_background

    a, b = shape[0] / 2, shape[1] / 2
    y, x = np.ogrid[-a:shape[0]-a, -b:shape[1]-b]
    v[x*x + y*y <= r*r] = vp

    return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                 dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)
