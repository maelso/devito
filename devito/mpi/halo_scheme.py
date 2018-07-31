from collections import OrderedDict
from itertools import product

from cached_property import cached_property

from devito.ir.support import Forward, Scope
from devito.logger import warning
from devito.parameters import configuration
from devito.types import LEFT, RIGHT
from devito.tools import Tag, as_mapper

__all__ = ['hs_build', 'HaloSchemeException']


class HaloLabel(Tag):
    pass


NONE = HaloLabel('none')
UNSUPPORTED = HaloLabel('unsupported')
STENCIL = HaloLabel('stencil')
FULL = HaloLabel('full')


class HaloScheme(object):

    """
    A HaloScheme describes a halo exchange pattern by means of two mappers:

        * ``fmapper``: ``Function -> [(Dimension, DataSide, amount), ...]``
        * ``fixed``: ``Function -> (Dimension -> Dimension)``

    ``fmapper`` tells the amount of data that :class:`TensorFunction`s should
    communicate along (a subset of) its :class:`Dimension`s.

    ``fixed`` tells how to access/insert the halo along the Dimensions
    where no halo exchange is performed. For example, consider the
    :class:`Function` ``u(t, x, y)``. Assume ``x`` and ``y`` require a halo
    exchange. The question is: once the halo exchange is performed, at what
    offset in ``t`` should it be placed? should it be at ``u(0, ...)`` or
    ``u(1, ...)`` or even ``u(t-1, ...)``? The ``fixed`` mapper provides
    this information.
    """

    def __init__(self):
        self._mapper = {}
        self._fixed = {}

    def __repr__(self):
        fnames = ",".join(i.name for i in set(self._mapper))
        return "HaloScheme<%s>" % fnames

    def add_halo_exchange(self, f, v):
        self._mapper.setdefault(f, []).append(v)

    def add_fixed_access(self, f, d, v):
        mapper = self._fixed.setdefault(f, {})
        if d in self._fixed[f]:
            raise ValueError("Redundant Dimension `%s` found" % d)
        mapper[d] = v

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


def hs_build(exprs, ispace, dspace):
    """
    Build a :class:`HaloScheme` for the :class:`TensorFunctions` in an
    iterable of expressions.

    :param exprs: The :class:`IREq`s for which the HaloScheme is built.
    :param ispace: A :class:`IterationSpace` describing the iteration
                   directions and the sub-iterators used within ``scope``.
    :param dspace: A :class:`DataSpace` describing the data access pattern
                   within ``scope``.
    """
    scope = Scope(exprs)

    hs_preprocess(scope)
    mapper = hs_classify(scope)
    hs = hs_compute(mapper, scope, ispace, dspace)

    return hs


def hs_preprocess(scope):
    """
    Perform some sanity checks to verify that it's actually possible/meaningful
    to derive a halo scheme for the given :class:`Scope`.
    """
    for i in scope.d_all:
        f = i.function
        if not f.is_TensorFunction:
            continue
        elif f.grid is None:
            raise HaloSchemeException("`%s` requires a `Grid`" % f.name)
        elif i.is_regular and any(f.grid.is_distributed(d) for d in i.cause):
            raise HaloSchemeException("`%s` is distributed, but is also used "
                                      "in a sequential iteration space" % i.cause)


def hs_classify(scope):
    """
    Return a mapper ``Function -> (Dimension -> [HaloLabel]`` describing what
    type of halo exchange is expected by the various :class:`TensorFunction`s
    in a :class:`Scope`.

    .. note::

        This function assumes as invariants all of the properties checked by
        :func:`hs_preprocess`.
    """
    mapper = {}
    for f, r in scope.reads.items():
        if not f.is_TensorFunction or f.grid is None:
            continue
        v = mapper.setdefault(f, {})
        for i in r:
            for d in i.findices:
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
                elif i.irregular(d) and f.grid.is_distributed(d):
                    v.setdefault(d, []).append(FULL)

    # Sanity check and reductions
    for f, v in mapper.items():
        for d, hl in list(v.items()):
            unique_hl = set(hl)
            if len(unique_hl) != 1:
                raise HaloSchemeException("Inconsistency found while building a halo "
                                          "scheme for `%s` along Dimension `%s`" % (f, d))
            v[d] = unique_hl.pop()

            if configuration['mpi'] and v[d] is UNSUPPORTED:
                warning("Distributed local-reductions over `%s` along "
                        "Dimension `%s` detected." % (f, d))

    return mapper


def hs_compute(mapper, scope, ispace, dspace):
    """
    Compute a halo scheme from a mapper as returned by :func:`hs_classify`,
    using the iteration space information carried by a :class:`IterationSpace`,
    the data access information carried by a :class:`DataSpace`, and the
    data dependence information carried by a :class:`Scope`.
    """
    hs = HaloScheme()
    for f, v in mapper.items():
        for d, hl in v.items():
            if hl is STENCIL:
                lower, upper = dspace[f][d.root].limits
                # Calculate what section of the halo region is needed, based on
                # the halo extent size and the stencil radius
                lsize = f._offset_domain[d].left - lower
                if lsize > 0:
                    hs.add_halo_exchange(f, (d, LEFT, lsize))
                rsize = upper - f._offset_domain[d].right
                if rsize > 0:
                    hs.add_halo_exchange(f, (d, RIGHT, rsize))
            elif hl is FULL:
                lsize = f._extent_halo[d].left
                if lsize > 0:
                    hs.add_halo_exchange(f, (d, LEFT, lsize))
                rsize = f._extent_halo[d].right
                if rsize > 0:
                    hs.add_halo_exchange(f, (d, RIGHT, rsize))
            elif hl is NONE:
                lower, upper = dspace[f][d.root].limits
                # There is no halo exchange along `d`, but we still need
                # to determine at what offset the halo will have be slotted in
                shift = int(any(d in i.cause for i in scope.d_all.project(f)))
                # Examples:
                # 1) u[t+1, x] = f(u[t, x])   => shift == 1
                # 2) u[t-1, x] = f(u[t, x])   => shift == 1
                # 3) u[t+1, x] = f(u[t+1, x]) => shift == 0
                # In the first and second cases, the x-halo should be inserted
                # at `t`, while in the last case it should be inserted at `t+1`
                if ispace.directions[d.root] is Forward:
                    last = upper - shift
                else:
                    last = lower + shift
                if d.is_Stepping:
                    # TimeFunctions using modulo-buffered iteration require
                    # special handling
                    subiters = ispace.sub_iterators.get(d.root, [])
                    submap = as_mapper(subiters, lambda md: md.modulo)
                    submap = {i.origin: i for i in submap[f._time_size]}
                    hs.add_fixed_access(f, d, submap[d + last])
                else:
                    hs.add_fixed_access(f, d, d.root + last)
    from IPython import embed; embed()
    return hs
