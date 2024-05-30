from functools import cached_property
from typing import Callable, Sequence, cast

import numpy as np
import porepy as pp
from porepy.models.momentum_balance import MomentumBalance


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = self.solid.convert_units(1.0, "m")
        y = self.solid.convert_units(1.0, "m")
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class RobinBoundaryConditionsWithBoundaryGrids:
    """Mixin for adaptations related to Robin boundary conditions with boundary grids.

    This mixin contains everything I needed to adapt in the source code for making the
    Robin boundary conditions (and thus absorbing boundary conditions (ABCs)) work with
    the boundary grid setup. Methods from three separate files are adapted. Which chunk
    of methods belong to which files are mentioned by a comment above the first method
    in the chunk.

    It also contains some brief "documentation" of other adaptations that were needed.
    Methods herein include:
    * _combine_boundary_operators: Signature now contains a robin_operator. The code
        within is adapted such that all three boundary operators are combined, not only
        Neumann and Dirichlet.
    * _update_bc_type_filter: Included a function for Robin analogous to the Neumann and
        Dirichlet ones. Now the Robin filter values can also be fetched from the parent
        grid, projected onto the boundary grid, and then updated.
    * __bc_type_storage: Needed to have this "locally". Not entirely sure why.
    * mechanical_stress: Adapt signature in call to _combine_boundary_operators to also
        give the Robin operator.
    * displacement_divergence: Adapt signature in call to _combine_boundary_operators
        to also give the Robin operator.
    * update_all_boundary_conditions: Include a call to update Robin boundary
        conditions.

    In addition to this one needs to define the robin boundary condition key (for
    identifying the operator/values/etc.) and a method for setting Robin-related
    boundary values.
    * self.bc_robin_key: "bc_robin".
    * self.bc_values_robin: Method for setting values to the Robin boundary conditions.
        That is, setting the right-hand side of sigma * n + alpha * u = G. Assigning the
        Robin weight has _not_ changed. This still happens in the bc_type_mechanics
        method.
        As of right now (while writing this docstring), the only occurence of this
        method is in runscripts utilizing Robin boundary conditions. This will be
        adapted soon.

    Specific change for the ABCs:
    * The right hand side of the ABCs is some coefficient multiplied by the previous
        face centered displacement value at the boundary. To obtain this, the utility
        method boundary_displacement (found in elastic_wave_equation_abc.py) was
        created to reconstruct the boundary displacements. Within here, a call to the
        (now deprecated) method bc_values_mechanics was found. Now we have to use the
        _combine_boundary_operators method instead.

    Changes related to non-trivial initial boundary values:
    * Certain simulations with ABCs include some non-trivial boundary values to be set.
        That is, a value that is dependent on the previous boundary displcement value.

        These values need to be initialized properly, and the way this was done before
        was to simply assign them before the simulation started. In the new set up,
        there is a check whether there are values present in the data dictionary the
        first time boundary conditions are to be updated. If initial values are set
        before this occurs, the method assumes the initial call has already been made
        and starts assigning new values and thus overriding the initial ones.

        This is solved within bc_values_robin by distinguishing what to return on the
        very first call to the method (self.time_manager.time_index == 0). This leads to
        the existence of an initial condition method specific for boundary values. The
        method itself is not too different from the previous method for setting initial
        bc values.

    """

    # Just needed
    @property
    def bc_robin_key(self) -> str:
        """The key for Robin boundary conditions."""
        return "bc_robin"

    @property
    def bc_values_mechanics_key(self) -> str:
        """Key for mechanical boundary conditions in the data dictionary."""
        return "bc_values_mechanics"

    # From boundary_condition.py
    def _combine_boundary_operators(
        self,
        subdomains: Sequence[pp.Grid],
        dirichlet_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        neumann_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        bc_type: Callable[[pp.Grid], pp.BoundaryCondition],
        name: str,
        robin_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        dim: int = 1,
    ) -> pp.ad.Operator:
        """Creates an operator representing Dirichlet and Neumann boundary conditions
        and projects it to the subdomains from boundary grids.

        Parameters:
            subdomains: List of subdomains.
            dirichlet_operator: Function that returns the Dirichlet boundary condition
                operator.
            neumann_operator: Function that returns the Neumann boundary condition
                operator.
            robin_operator: Function that returns the Robin boundary condition
                operator.
            dim: Dimension of the equation. Defaults to 1.
            name: Name of the resulting operator. Must be unique for an operator.

        Returns:
            Boundary condition representation operator.

        """
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)

        # Creating the Dirichlet, Neumann and Robin AD expressions.
        dirichlet = dirichlet_operator(boundary_grids)
        neumann = neumann_operator(boundary_grids)
        robin = robin_operator(boundary_grids)

        # Adding bc_type function to local storage to evaluate it before every time step
        # in case if the type changes in the runtime.
        self.__bc_type_storage[name] = bc_type
        # Creating the filters to ensure that Dirichlet, Neumann and Robin arrays do not
        # intersect where we do not want it.
        dir_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_dir"), domains=boundary_grids
        )
        neu_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_neu"), domains=boundary_grids
        )
        rob_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_rob"), domains=boundary_grids
        )
        # Setting the values of the filters for the first time.
        self._update_bc_type_filter(name=name, bc_type_callable=bc_type)

        boundary_to_subdomain = pp.ad.BoundaryProjection(
            self.mdg, subdomains=subdomains, dim=dim
        ).boundary_to_subdomain

        # Ensure that the Dirichlet operator only assigns (non-zero)
        # values to faces that are marked as having Dirichlet conditions.
        dirichlet *= dir_filter
        # Same with Neumann conditions.
        neumann *= neu_filter
        # Same with Robin conditions
        robin *= rob_filter
        # Projecting from the boundary grid to the subdomain.
        result = boundary_to_subdomain @ (dirichlet + neumann + robin)
        result.set_name(name)
        return result

    def _update_bc_type_filter(
        self, name: str, bc_type_callable: Callable[[pp.Grid], pp.BoundaryCondition]
    ):
        """Update the filters for Dirichlet, Neumann and Robin values.

        This is done to discard the data related to Dirichlet boundary condition in
        cells where the ``bc_type`` is Neumann or Robin and vice versa.

        """

        # Note: transposition is unavoidable to treat vector values correctly.
        def dirichlet(bg: pp.BoundaryGrid):
            # Transpose to get a n_face x nd array with shape compatible with
            # the projection matrix.
            is_dir = bc_type_callable(bg.parent).is_dir.T
            is_dir = bg.projection() @ is_dir
            # Transpose back, then ravel (in that order).
            return is_dir.T.ravel("F")

        def neumann(bg: pp.BoundaryGrid):
            is_neu = bc_type_callable(bg.parent).is_neu.T
            is_neu = bg.projection() @ is_neu
            return is_neu.T.ravel("F")

        def robin(bg: pp.BoundaryGrid):
            is_rob = bc_type_callable(bg.parent).is_rob.T
            is_rob = bg.projection() @ is_rob
            return is_rob.T.ravel("F")

        self.update_boundary_condition(name=(name + "_filter_dir"), function=dirichlet)
        self.update_boundary_condition(name=(name + "_filter_neu"), function=neumann)
        self.update_boundary_condition(name=(name + "_filter_rob"), function=robin)

    @cached_property
    def __bc_type_storage(self) -> dict[str, Callable[[pp.Grid], pp.BoundaryCondition]]:
        """Storage of functions that determine the boundary condition type on the given
        grid.

        Used in :meth:`update_all_boundary_conditions` for Dirichlet and Neumann
        filters.

        Stores per operator name (key) a callable (value) returning an operator
        representing the BC type per subdomain.

        """
        return {}

    # From constitutive_laws.py
    def mechanical_stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Linear elastic mechanical stress.

        .. note::
            The below discretization assumes the stress is discretized with a Mpsa
            finite volume discretization. Other discretizations may be possible, but are
            not available in PorePy at the moment, and would likely require changes to
            this method.

        Parameters:
            grids: List of subdomains or boundary grids. If subdomains, should be of
                co-dimension 0.

        Raises:
            ValueError: If any grid is not of co-dimension 0.
            ValueError: If any the method is called with a mixture of subdomains and
                boundary grids.

        Returns:
            Ad operator representing the mechanical stress on the faces of the grids.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            return self.create_boundary_operator(
                name=self.stress_keyword, domains=domains  # type: ignore[call-arg]
            )

        # Check that the subdomains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument subdomains a mixture of grids and
                                boundary grids"""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        for sd in domains:
            # The mechanical stress is only defined on subdomains of co-dimension 0.
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of co-dimension 0.")

        # No need to facilitate changing of stress discretization, only one is
        # available at the moment.
        discr = self.stress_discretization(domains)
        # Fractures in the domain
        interfaces = self.subdomains_to_interfaces(domains, [1])

        # Boundary conditions on external boundaries
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name=self.bc_values_mechanics_key,
            robin_operator=lambda bgs: self.create_boundary_operator(
                name=self.bc_robin_key, domains=bgs
            ),
        )

        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)
        # The stress in the subdomanis is the sum of the stress in the subdomain,
        # the stress on the external boundaries, and the stress on the interfaces.
        # The latter is found by projecting the displacement on the interfaces to the
        # subdomains, and let these act as Dirichlet boundary conditions on the
        # subdomains.
        stress = (
            discr.stress() @ self.displacement(domains)
            + discr.bound_stress() @ boundary_operator
            + discr.bound_stress()
            @ proj.mortar_to_primary_avg
            @ self.interface_displacement(interfaces)
        )
        stress.set_name("mechanical_stress")
        return stress

    def displacement_divergence(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Divergence of displacement [-].

        This is div(u). Note that opposed to old implementation, the temporal is not
        included here. Rather, it is handled by :meth:`pp.ad.dt`.

        Parameters:
            subdomains: List of subdomains where the divergence is defined.

        Returns:
            Divergence operator accounting from contributions from interior of the
            domain and from internal and external boundaries.

        """
        # Sanity check on dimension
        if not all(sd.dim == self.nd for sd in subdomains):
            raise ValueError("Displacement divergence only defined in nd.")

        # Obtain neighbouring interfaces
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Mock discretization (empty `discretize` method), used to access discretization
        # matrices computed by Biot discretization.
        discr = pp.ad.DivUAd(self.stress_keyword, subdomains, self.darcy_keyword)
        # Projections
        sd_projection = pp.ad.SubdomainProjections(subdomains, dim=self.nd)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=self.nd
        )

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name=self.bc_values_mechanics_key,
            robin_operator=lambda bgs: self.create_boundary_operator(
                name=self.bc_robin_key, domains=bgs
            ),
        )

        # Compose operator.
        div_u_integrated = discr.div_u() @ self.displacement(
            subdomains
        ) + discr.bound_div_u() @ (
            boundary_operator
            + sd_projection.face_restriction(subdomains)
            @ mortar_projection.mortar_to_primary_avg
            @ self.interface_displacement(interfaces)
        )
        # Divide by cell volumes to counteract integration.
        # The div_u discretization contains a volume integral. Since div u is used here
        # together with intensive quantities, we need to divide by cell volumes.
        cell_volumes_inv = pp.ad.Scalar(1) / self.wrap_grid_attribute(
            subdomains, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        div_u = cell_volumes_inv * div_u_integrated
        div_u.set_name("div_u")
        return div_u

    # From momentum_balance
    def update_all_boundary_conditions(self) -> None:
        """Set values for the displacement and the stress on boundaries."""
        super().update_all_boundary_conditions()
        self.update_boundary_condition(
            self.displacement_variable, self.bc_values_displacement
        )
        self.update_boundary_condition(self.stress_keyword, self.bc_values_stress)
        self.update_boundary_condition(self.bc_robin_key, self.bc_values_robin)


class BaseModel(
    MyGeometry,
    MomentumBalance,
): ...
