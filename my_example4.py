from devito import Grid, Function, TimeFunction
from devito.data.allocators import ExternalAllocator
import numpy as np

def grid_from_array(array, space_order=0):
    """
    Grid's shape plus borders length (2 times the space_order) 
    must match arrays's shape.

    Valid for 2D cases.
    """
    shape = tuple(x-2*y for x,y in zip(array.shape, (space_order,)*2))
    from devito import Grid
    return Grid(shape)


numpy_array_1 = np.array(np.mat('1 2; 3 4'), dtype=np.float32)
space_order=0
grid_2d_1 = grid_from_array(numpy_array_1,space_order)
f1 = Function(name='f1', grid=grid_2d_1, space_order=space_order,
             allocator=ExternalAllocator(numpy_array_1), initializer=lambda x: None)

f1.data[0,0] = 10
(numpy_array_1 == [[10., 2.],[3., 4.]]).all()

##
# This should not be accepted:
# AssertionError: Provided array has shape (2, 2). Expected (4, 4)
f2 = Function(name='f2', grid=grid_2d_1, space_order=1,
             allocator=ExternalAllocator(numpy_array_1), initializer=lambda x: None)
##

grid_2d_dev = Grid(shape=(2, 2))
f3 = Function(name='f3', grid=grid_2d_dev, space_order=1)



numpy_array_2 = np.array(np.mat('1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16'), 
                         dtype=np.float32)
space_order=1
grid_2d_2 = grid_from_array(numpy_array_2, space_order)
f4 = Function(name='f4', grid=grid_2d_2, space_order=space_order,
             allocator=ExternalAllocator(numpy_array_2), initializer=lambda x: None)

f4.data[0,1]=100
f4._data[0,1]=101

(numpy_array_2 == [[  1., 101.,   3.,   4.],
                   [  5.,   6., 100.,   8.],
                   [  9.,  10.,  11.,  12.],
                   [ 13.,  14.,  15.,  16.]]).all()
