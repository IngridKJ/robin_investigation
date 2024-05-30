import porepy as pp
import numpy as np

from model import BaseModel


class BoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """"""
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "dir",
        )
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        displacement_values = np.zeros((self.nd, bg.num_cells))
        values[0][bounds.west] += np.ones(len(displacement_values[0][bounds.west]))

        return values.ravel("F")


class Model(BoundaryConditions, BaseModel): ...


params = {"folder_name": "all_dirichlet"}
model = Model(params)
pp.run_time_dependent_model(model, params)
