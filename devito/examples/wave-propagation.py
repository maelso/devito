from devito import TimeFunction, Eq, solve, Operator, configuration
from examples.seismic import Model, plot_velocity, plot_image, TimeAxis, RickerSource
import numpy as np

configuration['jit-backdoor'] = False

# Define a physical size
size = 16384
shape = (size, size)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m.
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :] = 1.5

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=2)
# plot_velocity(model)

t0 = 0.  # Simulation starts a t=0
tn = 18000.  # Simulation last 18 second8 (18000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)

# We can now write the PDE
pde = model.m * u.dt2 - u.laplace
stencil = Eq(u.forward, solve(pde, u.forward))

# Finally we define the source injection in the center of the model
u.data[0][len(u.data[0])//2][len(u.data[0])//2] = 1.

op = Operator([stencil], subs=model.spacing_map)

print(op)

op.apply(time=time_range.num-1, dt=model.critical_dt)

plot_image(u.data[0])
