from devito import TimeFunction, Eq, solve, Operator
from examples.seismic import Model, plot_velocity, plot_image, TimeAxis, RickerSource
import numpy as np

# Define a physical size
shape = (64, 64)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 2.5

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=2)
# plot_velocity(model)

t0 = 0.  # Simulation starts a t=0
tn = 400.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)
# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 20.  # Depth is 20m
# We can plot the time signature to see the wavelet
# src.show()

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
from IPython import embed;embed()
# We can now write the PDE
pde = model.m * u.dt2 - u.laplace 
stencil = Eq(u.forward, solve(pde, u.forward))

# Finally we define the source injection and receiver read function to generate the corresponding code
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)

op = Operator([stencil] + src_term, subs=model.spacing_map, dse='aggressive')

print(op)
op.apply(time=time_range.num-1, dt=model.critical_dt)

print(u.data[0])


plot_image(u.data[0])