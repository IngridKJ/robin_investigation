import porepy as pp
import numpy as np

from model import BaseModelRobin


class BoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """"""
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )
        # Only the eastern boundary will be Robin (absorbing)
        bc.is_rob[:, bounds.west] = False

        # Western boundary is Dirichlet
        bc.is_dir[:, bounds.west] = True
        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def assign_robin_weight(
        self, sd: pp.Grid, bc: pp.BoundaryConditionVectorial
    ) -> None:
        """"""
        # Initiating the arrays for the Robin weight
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")
        bc.robin_weight = value * self.weight_coefficient_for_testing

    def bc_values_robin(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """"""
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        displacement_values = np.zeros((self.nd, bg.num_cells))
        values[0][bounds.west] += np.ones(len(displacement_values[0][bounds.west]))

        return values.ravel("F")


class Model(BoundaryConditions, BaseModelRobin): ...


# coefficients = [
#     (0, "alpha_0", "simplex"),
#     (1, "alpha_1", "simplex"),
#     (0.000001, "alpha_0_000001", "simplex"),
#     (5e12, "alpha_5e12", "simplex"),
# ]
coefficients = [
    (0, "alpha_0", "cartesian"),
    (1, "alpha_1", "cartesian"),
    (0.000001, "alpha_0_000001", "cartesian"),
    (5e12, "alpha_5e12", "cartesian"),
]

for coefficient in coefficients:
    params = {
        "folder_name": f"west_dirichlet_{str(coefficient[1])}_{coefficient[2]}",
        "grid_type": coefficient[2],
    }
    model = Model(params)
    model.weight_coefficient_for_testing = coefficient[0]
    pp.run_time_dependent_model(model, params)
