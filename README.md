# robin_investigation
Investigating the Robin conditions in PorePy.

# Setup
All the simulations have a west Dirichlet boundary of value 1. 
The other boundaries are either:
* All zero Dirichlet
* All zero Neumann
* All zero Robin in the limit case Dirichlet or Neumann

The Robin conditions are on the form: sigma * n + alpha * u = G. 
The limit case where alpha goes to infinity is a Dirichlet condition, and when alpha goes
to zero it is a Neumann condition.

The intention is to check if the Robin condition implementation honors these limit cases
as expected. 
* In the case of a huge alpha (I chose 5e12), we should see similar simulation results
  as when Dirichlet conditions are chosen for the remaining boundaries. See [here for
  simplex](./robin_limit_to_dirichlet_alpha_5e12.png) and [here for
  cartesian](./robin_limit_to_dirichlet_alpha_5e12_cartesian.png) grid.
* In the case of a small alpha (I chose alpha = 0 for simplex grids, and alpha = 0 and
  alpha = 0.000001 for cartesian grids), we should see similar simulation results as
  when Neumann are chosen for the remaining boundaries. Alpha = 0 worked nicely for
  simplex grids, and not as well for cartesian. See [here for
  simplex](./robin_limit_to_neumann_alpha_0_cartesian.png), [here for
  cartesian (alpha = 0)](./robin_limit_to_neumann_alpha_0_cartesian.png)  and [here for
  cartesian (alpha = 0.000001)](./robin_limit_to_neumann_alpha_0_000001_cartesian.png).
