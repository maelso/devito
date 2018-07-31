from collections import OrderedDict
from itertools import product

from cached_property import cached_property

from devito.ir.support import Forward
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
    exchange. A question is: once the halo exchange is performed, at what
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


def hs_build(ispace, dspace, scope):
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
    hs_preprocess(ispace, scope)
    mapper = hs_classify(ispace, scope)
    hs = hs_compute(mapper, ispace, dspace, scope)

    return hs


def hs_preprocess(ispace, scope):
    """
    Perform some sanity checks to verify that it's actually possible/meaningful
    to derive a halo scheme for the given :class:`IterationSpace` and :class:`Scope`.
    """
    for d in ispace.dimensions:
        for i in scope.d_all:
            f = i.function
            if not f.is_TensorFunction:
                continue
            elif f.grid is None:
                raise HaloSchemeException("`%s` requires a `Grid`" % f.name)
            elif f.grid.is_distributed(d):
                if i.is_regular:
                    if d in i.cause:
                        raise HaloSchemeException("`%s` is distributed, but is also used "
                                                  "in a sequential iteration space" % d)
                else:
                    raise HaloSchemeException("`%s` is distributed, so it cannot be used "
                                              "to define irregular iteration space" % d)
            elif not (i.is_carried(d) or i.is_increment):
                raise HaloSchemeException("Cannot derive a halo scheme due to the "
                                          "irregular data dependence `%s`" % i)


def hs_classify(ispace, scope):
    """
    Return a mapper ``Function -> (Dimension -> [HaloLabel]`` describing what
    type of halo exchange is expected by the various :class:`TensorFunction`s
    within a :class:`Scope` before executing an :class:`IterationSpace`.

    .. note::

        This function assumes as invariants all of the properties checked by
        :func:`hs_preprocess`.
    """
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


def hs_compute(mapper, ispace, dspace, scope):
    """
    Compute a halo scheme from a mapper as returned by :func:`hs_classify`,
    using the iteration space information carried by a :class:`IterationSpace`,
    the data access information carried by a :class:`DataSpace`, and the
    data dependence information carried by a :class:`Scope`.
    """
    hs = HaloScheme()
    for f, v in mapper.items():
        for d, hl in v.items():
            lower, upper = dspace[f][d.root].limits
            if hl is STENCIL:
                # Calculate what section of the halo region is needed, based on
                # the halo extent size and the stencil radius
                lsize = f._offset_domain[d].left - lower
                if lsize > 0:
                    hs.add_halo_exchange(f, (d, LEFT, lsize))
                rsize = upper - f._offset_domain[d].right
                if rsize > 0:
                    hs.add_halo_exchange(f, (d, RIGHT, rsize))
            elif hl is NONE:
                # There is no halo exchange along `d`, but we still need
                # to determine at what offset the halo will have be slotted in
                shift = int(any(d in i.cause for i in scope.project(f)))
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
                    hs.add_fixed_access(f, d, d + submap[d + last])
                else:
                    hs.add_fixed_access(f, d, d.root + last)
    from IPython import embed; embed()
    return hs
