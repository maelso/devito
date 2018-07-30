from collections import OrderedDict
from itertools import product

from cached_property import cached_property

from devito.ir.support import Scope, Forward
from devito.logger import warning
from devito.parameters import configuration
from devito.types import OWNED, HALO, LEFT, RIGHT
from devito.tools import Tag, as_mapper

__all__ = ['get_views', 'HaloScheme', 'HaloSchemeException', 'derive_halo_scheme']


def get_views(f, fixed):
    """
    Return a mapper ``(dimension, side, region) -> (size, offset)`` for a
    :class:`TensorFunction`.
    """
    mapper = OrderedDict()
    for dimension, side, region in product(f.dimensions, [LEFT, RIGHT], [OWNED, HALO]):
        if dimension in fixed:
            continue
        sizes = []
        offsets = []
        for d, i in zip(f.dimensions, f.symbolic_shape):
            if d in fixed:
                offsets.append(fixed[d])
            elif dimension is d:
                offset, extent = f._get_region(region, dimension, side, True)
                sizes.append(extent)
                offsets.append(offset)
            else:
                sizes.append(i)
                offsets.append(0)
        mapper[(dimension, side, region)] = (sizes, offsets)
    return mapper


class HaloLabel(Tag):
    pass
NONE = HaloLabel('none')  # noqa
UNSUPPORTED = HaloLabel('unsupported')
STENCIL = HaloLabel('stencil')
FULL = HaloLabel('full')


def derive_halo_scheme(ispace, dspace, scope):
    """
    Derive a halo exchange scheme describing, for given iteration and
    data spaces:

        * what :class:`Function`s may require a halo update before executing
          the iteration space,
        * and what values of such :class:`Function`s should actually be exchanged

    :param ispace: A :class:`IterationSpace` for which a halo scheme is built.
    :param dspace: A :class:`DataSpace` describing the data access pattern
                   within ``ispace``.
    :param scope: A :class:`Scope` describing the data dependence pattern
                  within ``ispace``.
    """
    # 1) Sanity checks -- is it actually possible/meaningful to derive a halo scheme?
    for d in ispace.dimensions:
        for dep in scope.d_all:
            f = dep.function
            if not f.is_TensorFunction:
                continue
            elif f.grid is None:
                raise HaloSchemeException("`%s` requires a `Grid`" % f.name)
            elif f.grid.is_distributed(d):
                if dep.is_regular:
                    if d in dep.cause:
                        raise HaloSchemeException("`%s` is distributed, but is also used "
                                                  "in a sequential iteration space" % d)
                else:
                    raise HaloSchemeException("`%s` is distributed, so it cannot be used "
                                              "to define irregular iteration space" % d)
            elif not (dep.is_carried(d) or dep.is_increment):
                raise HaloSchemeException("Cannot derive a halo scheme due to the "
                                          "irregular data dependence `%s`" % dep)

    # 2) Find out what functions/dimensions need a halo exchange
    mapper = {}
    for d in ispace.dimensions:
        for f, r in scope.reads.items():
            if not f.is_TensorFunction or f.grid is None:
                continue
            v = mapper.setdefault(f, {})
            for i in r:
                if i.affine(d):
                    if f.grid.is_distributed(d):
                        v.setdefault(d, []).append(STENCIL)
                    else:
                        v.setdefault(d, []).append(NONE)
                elif i.is_increment:
                    # A read used for a distributed local-reduction. Users are expected
                    # to deal with this data access pattern by themselves, for example
                    # by resorting to common techniques such as redundant computation
                    v.setdefault(d, []).append(UNSUPPORTED)
                elif i.is_irregular:
                    # We need to find out what data dimensions `d` appears in
                    # so that, conservatively, we can exchange all these halos
                    for ai, fi in zip(i.aindices, i.findices):
                        if f.grid.is_distributed(fi) and ({None, d} & {ai}):
                            v.setdefault(fi, []).append(FULL)
    from IPython import embed; embed()

    #if configuration['mpi']:
    #    warning("Local distributed reductions over `%s` detected.")
    #    continue


class HaloScheme(object):

    """
    A HaloScheme consists of several mappers describing a halo exchange pattern.

        * ``dmapper``: ``dimension -> [(function, side, amount), ...]``
        * ``fmapper``: ``function -> [(dimension, side, amount), ...]``

    For the :class:`Dimension`s that need no halo exchange but appear in one
    or more of the tracked :class:`TensorFunction`s, a further mapper

        * ``fixed``: ``dimension -> dimension``

    is provided telling at which index in these dimensions should the halo
    be inserted.
    For example, consider the :class:`Function` ``u(t, x, y)``. Assume ``x`` and
    ``y`` require a halo exchange. A question is: once the halo exchange is
    performed, at what offset in ``t`` should it be placed? should it be at
    ``u(0, ...)`` or ``u(1, ...)`` or even ``u(t-1, ...)``? The ``fixed`` mapper
    provides this information.
    """

    def __init__(self, exprs):
        self._mapper = {}
        self._fixed = {}

        # What Functions actually need a halo exchange?
        need_halo = as_mapper(Scope(exprs).d_all, lambda i: i.function)
        need_halo = {k: v for k, v in need_halo.items() if k.is_TensorFunction}

        for i in exprs:
            for f, v in i.dspace.parts.items():
                if f not in need_halo:
                    continue
                if f.grid is None:
                    raise RuntimeError("`%s` needs a `Grid` for a HaloScheme" % f.name)
                for d in f.dimensions:
                    r = d.root
                    if v[r].is_Null:
                        continue
                    elif d in f.grid.distributor.dimensions:
                        # Found a distributed dimension, calculate what and how
                        # much halo is needed
                        lsize = f._offset_domain[d].left - v[r].lower
                        if lsize > 0:
                            self._mapper.setdefault(f, []).append((d, LEFT, lsize))
                        rsize = v[r].upper - f._offset_domain[d].right
                        if rsize > 0:
                            self._mapper.setdefault(f, []).append((d, RIGHT, rsize))
                    else:
                        # Found a serial dimension, we need to determine where,
                        # along this dimension, the halo will have to be placed
                        fixed = self._fixed.setdefault(f, OrderedDict())
                        shift = int(any(d in dep.cause for dep in need_halo[f]))
                        # Examples:
                        # u[t+1, x] = f(u[t, x])   => shift == 1
                        # u[t-1, x] = f(u[t, x])   => shift == 1
                        # u[t+1, x] = f(u[t+1, x]) => shift == 0
                        # In the first and second cases, the x-halo should be inserted
                        # at `t`, while in the last case it should be inserted at `t+1`
                        if i.ispace.directions[r] is Forward:
                            last = v[r].upper - shift
                        else:
                            last = v[r].lower + shift
                        if d.is_Stepping:
                            # TimeFunctions using modulo-buffered iteration require
                            # special handling
                            subiters = i.ispace.sub_iterators.get(r, [])
                            submap = as_mapper(subiters, lambda md: md.modulo)
                            submap = {i.origin: i for i in submap[f._time_size]}
                            try:
                                handle = submap[d + last]
                            except KeyError:
                                raise HaloSchemeException
                        else:
                            handle = r + last
                        if handle is not None and handle != fixed.get(d, handle):
                            raise HaloSchemeException
                        fixed[d] = handle or fixed.get(d)

    @property
    def fixed(self):
        return self._fixed

    @property
    def fmapper(self):
        return self._mapper

    @cached_property
    def dmapper(self):
        mapper = {}
        for f, v in self.fmapper.items():
            for d, side, size in v:
                mapper.setdefault(d, []).append((f, side, size))
        return mapper

    @cached_property
    def mask(self):
        mapper = {}
        for f, v in self.fmapper.items():
            needed = [(d, side) for d, side, _ in v]
            for i in product(f.dimensions, [LEFT, RIGHT]):
                if i[0] in self.fixed.get(f, []):
                    continue
                mapper.setdefault(f, OrderedDict())[i] = i in needed
        return mapper


class HaloSchemeException(Exception):
    pass
