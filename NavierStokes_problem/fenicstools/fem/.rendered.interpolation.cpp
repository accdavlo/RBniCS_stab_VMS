/*

*/
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Expression.h>
#include <dolfin/fem/FiniteElement.h>

#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/common/RangedIndexSet.h>

using namespace dolfin;
namespace py = pybind11;

// Comparison operator for hashing coordinates. Note that two
// coordinates are considered equal if equal to within specified
// tolerance.
struct lt_coordinate
{
  lt_coordinate(double tolerance) : TOL(tolerance) {}

  bool operator() (const std::vector<double>& x,
                   const std::vector<double>& y) const
  {
    std::size_t n = std::max(x.size(), y.size());
    for (std::size_t i = 0; i < n; ++i)
    {
      double xx = 0.0;
      double yy = 0.0;
      if (i < x.size())
        xx = x[i];
      if (i < y.size())
        yy = y[i];

      if (xx < (yy - TOL))
        return true;
      else if (xx > (yy + TOL))
        return false;
    }
    return false;
  }

  // Tolerance
  const double TOL;
};

void extract_dof_component_map(std::unordered_map<std::size_t,
                               std::size_t>& dof_component_map,
                               const FunctionSpace& V,
                               int* component)
{
  // Extract sub dofmaps recursively and store dof to component map
  if (V.element()->num_sub_elements() == 0)
  {
    std::unordered_map<std::size_t, std::size_t> collapsed_map;
    std::shared_ptr<GenericDofMap> dummy
      = V.dofmap()->collapse(collapsed_map, *V.mesh());
    (*component)++;
    for (const auto &map_it : collapsed_map)
      dof_component_map[map_it.second] = (*component);
  }
  else
  {
    for (std::size_t i = 0; i < V.element()->num_sub_elements(); ++i)
    {
      const std::vector<std::size_t> comp = {i};
      std::shared_ptr<FunctionSpace> Vs = V.extract_sub_space(comp);
      extract_dof_component_map(dof_component_map, *Vs, component);
    }
  }}

bool in_bounding_box(const std::vector<double>& point,
                     const std::vector<double>& bounding_box,
                     const double tol)
{
  // Return false if bounding box is empty
  if (bounding_box.empty())
    return false;

  const std::size_t gdim = point.size();
  dolfin_assert(bounding_box.size() == 2*gdim);
  for (std::size_t i = 0; i < gdim; ++i)
  {
    if (!(point[i] >= (bounding_box[i] - tol)
          && point[i] <= (bounding_box[gdim + i] + tol)))
    {
      return false;
    }
  }
  return true;}

std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
tabulate_coordinates_to_dofs(const FunctionSpace& V)
{
    std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    coords_to_dofs(lt_coordinate(1.0e-12));

  // Extract mesh, dofmap and element
  dolfin_assert(V.dofmap());
  dolfin_assert(V.element());
  dolfin_assert(V.mesh());
  const GenericDofMap& dofmap = *V.dofmap();
  const FiniteElement& element = *V.element();
  const Mesh& mesh = *V.mesh();

  // Geometric dimension
  const std::size_t gdim = mesh.geometry().dim();

  // Loop over cells and tabulate dofs
  boost::multi_array<double, 2> coordinates;
  std::vector<double> coordinate_dofs;
  std::vector<double> coors(gdim);

  // Speed up the computations by only visiting (most) dofs once
  const std::size_t local_size = dofmap.ownership_range().second
    - dofmap.ownership_range().first;
  RangedIndexSet already_visited(std::make_pair(0, local_size));

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    cell->get_coordinate_dofs(coordinate_dofs);

    // Get local-to-global map
    auto dofs = dofmap.cell_dofs(cell->index());

    // Tabulate dof coordinates on cell
    element.tabulate_dof_coordinates(coordinates, coordinate_dofs,
                                     *cell);

    // Map dofs into coords_to_dofs
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
    {
      const std::size_t dof = dofs[i];
      if (dof < local_size)
      {
        // Skip already checked dofs
        if (!already_visited.insert(dof))
          continue;

        // Put coordinates in coors
        std::copy(coordinates[i].begin(), coordinates[i].end(),
                  coors.begin());

        // Add dof to list at this coord
        const auto ins = coords_to_dofs.insert
          (std::make_pair(coors, std::vector<std::size_t>{dof}));
        if (!ins.second)
          ins.first->second.push_back(dof);
      }
    }
  }
  return coords_to_dofs;
}

void interpolate1(const Expression& u0, Function& u)
{
  // Get function space and element interpolating to
  dolfin_assert(u.function_space());
  const FunctionSpace& V = *u.function_space();
  dolfin_assert(V.element());
  const FiniteElement& element = *V.element();

  // Check that function ranks match
  if (element.value_rank() != u0.value_rank())
  {
      dolfin_error("interpolate.cpp",
                  "interpolate Expression into function space",
                  "Rank of Expression (%d) does not match rank of function space (%d)",
                  u0.value_rank(), element.value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < element.value_rank(); ++i)
  {
      if (element.value_dimension(i) != u0.value_dimension(i))
      {
      dolfin_error("interpolate.cpp",
                  "interpolate Expression into function space",
                  "Dimension %d of Expression (%d) does not match dimension %d of function space (%d)",
                  i, u0.value_dimension(i), i, element.value_dimension(i));
      }
  }

  // Get mesh and dimension of FunctionSpace interpolating to
  dolfin_assert(V.mesh());
  const Mesh& mesh = *V.mesh();
  const std::size_t gdim = mesh.geometry().dim();

  // Create arrays used to evaluate one point
  std::vector<double> x(gdim);
  std::vector<double> values(u0.value_size());
  Array<double> _x(gdim, x.data());
  Array<double> _values(u0.value_size(), values.data());

  // Create vector to hold all local values of u
  std::vector<double> local_u_vector(u.vector()->local_size());

  // Create map from coordinates to dofs sharing that coordinate
  const std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
      coords_to_dofs = tabulate_coordinates_to_dofs(V);

  // Get a map from global dofs to component number in mixed space
  std::unordered_map<std::size_t, std::size_t> dof_component_map;
  int component = -1;
  extract_dof_component_map(dof_component_map, V, &component);

  // Evaluate all points
  for (const auto &map_it : coords_to_dofs)
  {
      // Place interpolation point in x
      std::copy(map_it.first.begin(), map_it.first.end(), x.begin());

      u0.eval(_values, _x);
      for (const auto &d : map_it.second)
      {
      dolfin_assert(d < local_u_vector.size());
      local_u_vector[d] = values[dof_component_map[d]];
      }
  }

  // Set and finalize
  u.vector()->set_local(local_u_vector);
  u.vector()->apply("insert");
}

void interpolate2(const Function& u0, Function& u)
{
  // Interpolate from Function u0 to Function u.
  // This mesh of u0 may be different from that of u
  //
  // The algorithm is briefly
  //
  //   1) Create a map from all different coordinates of u's dofs to
  //      the dofs living on that coordinate. This is done such that
  //      one only need to visit (and distribute) each interpolation
  //      point once.
  //   2) Create a map from dof to component index in Mixed Space.
  //   3) Create bounding boxes for the partitioned mesh of u0 and
  //      distribute to all processors.
  //   4) Using bounding boxes, compute the processes that *may* own
  //      the dofs of u.
  //   5) Distribute interpolation points to potential owners who
  //      subsequently tries to evaluate u0. If successful, return
  //      values of u0 to owner.

  // Get function spaces of Functions interpolating to/from
  dolfin_assert(u0.function_space());
  dolfin_assert( u.function_space());
  const FunctionSpace& V0 = *u0.function_space();
  const FunctionSpace& V1 =  *u.function_space();

  // Get element interpolating to
  dolfin_assert(V1.element());
  const FiniteElement& element = *V1.element();

  // Check that function ranks match
  if (element.value_rank() != u0.value_rank())
  {
    dolfin_error("LagrangeInterpolator.cpp",
                 "interpolate Function into function space",
                 "Rank of Function (%d) does not match rank of function space (%d)",
                 u0.value_rank(), element.value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < element.value_rank(); ++i)
  {
    if (element.value_dimension(i) != u0.value_dimension(i))
    {
      dolfin_error("LagrangeInterpolator.cpp",
                   "interpolate Function into function space",
                   "Dimension %d of Function (%d) does not match dimension %d of function space (%d)",
                   i, u0.value_dimension(i), i, element.value_dimension(i));
    }
  }

  // Get mesh and dimension of FunctionSpace interpolating to/from
  dolfin_assert(V0.mesh());
  dolfin_assert(V1.mesh());
  const Mesh& mesh0 = *V0.mesh();
  const Mesh& mesh1 = *V1.mesh();
  const std::size_t gdim0 = mesh0.geometry().dim();
  const std::size_t gdim1 = mesh1.geometry().dim();

  // Get communicator
  const MPI_Comm mpi_comm = V1.mesh()->mpi_comm();

  // Create bounding box of mesh0
  std::vector<double> x_min_max(2*gdim0);
  std::vector<double> coordinates = mesh0.coordinates();
  for (std::size_t i = 0; i < gdim0; ++i)
  {
    for (auto it = coordinates.begin() + i; it < coordinates.end(); it += gdim0)
    {
      x_min_max[i]         = std::min(x_min_max[i], *it);
      x_min_max[gdim0 + i] = std::max(x_min_max[gdim0 + i], *it);
    }
  }

  // Communicate bounding boxes
  std::vector<std::vector<double>> bounding_boxes;
  dolfin::MPI::all_gather(mpi_comm, x_min_max, bounding_boxes);

  // Create arrays used to evaluate one point
  std::vector<double> x(gdim0);
  std::vector<double> values(u0.value_size());
  Array<double> _x(gdim0, x.data());
  Array<double> _values(u0.value_size(), values.data());

  // Create vector to hold all local values of u
  std::vector<double> local_u_vector(u.vector()->local_size());

  // Create map from coordinates to dofs sharing that coordinate
  std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    coords_to_dofs = tabulate_coordinates_to_dofs(V1);

  // Get a map from global dofs to component number in mixed space
  std::unordered_map<std::size_t, std::size_t> dof_component_map;
  int component = -1;
  extract_dof_component_map(dof_component_map, V1, &component);

  // Search this process first for all coordinates in u's local mesh
  std::vector<double> points_not_found;
  for (const auto &map_it : coords_to_dofs)
  {
    // Place interpolation point in x
    std::copy(map_it.first.begin(), map_it.first.end(), x.begin());

    try
    { // Store values when point is found
      u0.eval(_values, _x);
      std::vector<std::size_t> dofs = map_it.second;
      for (const auto &d : map_it.second)
        local_u_vector[d] = values[dof_component_map[d]];
    }
    catch (std::exception &e)
    {
      // If not found then it must be searched on the other processes
      points_not_found.insert(points_not_found.end(), x.begin(), x.end());
    }
  }

  // Get number of MPI processes
  std::size_t num_processes = dolfin::MPI::size(mpi_comm);

  // Remaining interpolation points must be found through MPI
  // communication.  Check first using bounding boxes which process
  // may own the points
  std::vector<std::vector<double>> potential_points(num_processes);
  for (std::size_t i = 0; i < points_not_found.size(); i += gdim1)
  {
    std::copy(points_not_found.begin() + i,
              points_not_found.begin() + i + gdim1, x.begin());

    // Find potential owners
    for (std::size_t p = 0; p < num_processes; p++)
    {
      if (p == dolfin::MPI::rank(mpi_comm))
        continue;

      // Check if in bounding box
      if (in_bounding_box(x, bounding_boxes[p], 1e-12))
      {
        potential_points[p].insert(potential_points[p].end(),
                                   x.begin(), x.end());
      }
    }
  }

  // Communicate all potential points
  std::vector<std::vector<double>> potential_points_recv;
  dolfin::MPI::all_to_all(mpi_comm, potential_points, potential_points_recv);

  // Now try to eval u0 for the received points
  std::vector<std::vector<double>> coefficients_found(num_processes);
  std::vector<std::vector<double>> points_found(num_processes);

  for (std::size_t p = 0; p < num_processes; ++p)
  {
    if (p == dolfin::MPI::rank(mpi_comm))
      continue;

    std::vector<double>& points = potential_points_recv[p];
    for (std::size_t j = 0; j < points.size()/gdim1; ++j)
    {
      std::copy(points.begin() + j*gdim1, points.begin() + (j + 1)*gdim1,
                x.begin());

      try
      {
        // push back when point is found
        u0.eval(_values, _x);
        coefficients_found[p].insert(coefficients_found[p].end(),
                                     values.begin(), values.end());
        points_found[p].insert(points_found[p].end(), x.begin(), x.end());
      }
      catch (std::exception &e)
      {
        // If not found then do nothing
      }
    }
  }

  // Send back the found coefficients and points
  std::vector<std::vector<double>> coefficients_recv;
  std::vector<std::vector<double>> points_recv;
  dolfin::MPI::all_to_all(mpi_comm, coefficients_found, coefficients_recv);
  dolfin::MPI::all_to_all(mpi_comm, points_found, points_recv);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    if (p == dolfin::MPI::rank(mpi_comm))
      continue;

    // Get the new values and points
    const std::vector<double>& vals = coefficients_recv[p];
    const std::vector<double>& pts = points_recv[p];

    // Move all found coefficients into the local_u_vector
    for (std::size_t j = 0; j < pts.size()/gdim1; ++j)
    {
      std::copy(pts.begin() + j*gdim1, pts.begin() + (j + 1)*gdim1, x.begin());

      // Get the owned dofs sharing x
      const std::vector<std::size_t>& dofs = coords_to_dofs[x];

      // Place result in local_u_vector
      for (const auto &d : dofs)
      {
        dolfin_assert(d <  local_u_vector.size());
        local_u_vector[d]
          = vals[j*u0.value_size() + dof_component_map[d]];
      }
    }
  }

  // Set and finalize
  u.vector()->set_local(local_u_vector);
  u.vector()->apply("insert");
  }

void interpolate_any1(const Function& u0, Function& u)
{
  // Interpolate from Function u0 to Function u.
  // This mesh of u0 may be different from that of u
  //
  // The algorithm is briefly
  //
  //   1) Create a set of all different coordinates of u's that will
  //      require an eval (like tabulate_all_coordinates)
  //   2) Create global bounding boxes for the partitioned mesh of u0 and
  //      distribute to all processors.
  //   3) Using bounding boxes, compute the processes that *may* own
  //      the points in need of eval.
  //   4) Distribute interpolation points to potential owners who
  //      subsequently tries to evaluate u0. If successful, return
  //      values (result of eval) of u0 to owner.
  //   5) Wrap the results of the global function evals in an Expression.
  //      Now this Expression contains all global eval results that will
  //      be required
  //   6) Call evaluate_dofs for all cells on local mesh using the
  //      wrapped function as the ufc function.
  //   7) Update coefficients of local vector with results from
  //      evaluate_dofs

  // Get function spaces of Functions interpolating to/from
  dolfin_assert(u0.function_space());
  dolfin_assert( u.function_space());
  const FunctionSpace& V0 = *u0.function_space();
  const FunctionSpace& V1 =  *u.function_space();

  // Get element interpolating to
  dolfin_assert(V1.element());
  const FiniteElement& element = *V1.element();

  // Check that function ranks match
  if (element.value_rank() != u0.value_rank())
  {
      dolfin_error("LagrangeInterpolator.cpp",
                  "interpolate Function into function space",
                  "Rank of Function (%d) does not match rank of function space (%d)",
                  u0.value_rank(), element.value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < element.value_rank(); ++i)
  {
      if (element.value_dimension(i) != u0.value_dimension(i))
      {
      dolfin_error("LagrangeInterpolator.cpp",
                  "interpolate Function into function space",
                  "Dimension %d of Function (%d) does not match dimension %d of function space (%d)",
                  i, u0.value_dimension(i), i, element.value_dimension(i));
      }
  }

  // Get mesh and dimension of FunctionSpace interpolating to/from
  dolfin_assert(V0.mesh());
  dolfin_assert(V1.mesh());
  const Mesh& mesh0 = *V0.mesh();
  const Mesh& mesh1 = *V1.mesh();
  const std::size_t gdim0 = mesh0.geometry().dim();
  const std::size_t gdim1 = mesh1.geometry().dim();

  // Get communicator
  const MPI_Comm mpi_comm = V1.mesh()->mpi_comm();

  // Create bounding box of mesh0
  std::vector<double> x_min_max(2*gdim0);
  std::vector<double> coordinates = mesh0.coordinates();
  for (std::size_t i = 0; i < gdim0; ++i)
  {
      for (auto it = coordinates.begin() + i; it < coordinates.end(); it += gdim0)
      {
      x_min_max[i]         = std::min(x_min_max[i], *it);
      x_min_max[gdim0 + i] = std::max(x_min_max[gdim0 + i], *it);
      }
  }

  // Communicate bounding boxes
  std::vector<std::vector<double>> bounding_boxes;
  dolfin::MPI::all_gather(mpi_comm, x_min_max, bounding_boxes);

  // Create arrays used to evaluate one point
  std::vector<double> x(gdim0);
  std::vector<double> values(u0.value_size());
  Array<double> _x(gdim0, x.data());
  Array<double> _values(u0.value_size(), values.data());

  // Create vector to hold all local values of u
  std::vector<double> local_u_vector(u.vector()->local_size());

  // Get dofmap of u
  dolfin_assert(V1.dofmap());
  const GenericDofMap& dofmap = *V1.dofmap();

  // Create map from coordinate to global result of eval
  static std::map<std::vector<double>, std::vector<double>, lt_coordinate>
      coords_to_values(lt_coordinate(1.0e-12));

  // Create an Expression wrapping the set of coordinates
  // Objective is to obtain all points in need of evaluation.
  static std::set<std::vector<double>> coords;
  class CordFunction : public Expression
  {
  public:
      CordFunction(int value_shape) : Expression(value_shape) {};
      mutable std::vector<double> _xx;
      void eval(Array<double>& values, const Array<double>& x) const
      {
      for (uint j = 0; j < x.size(); j++)
          _xx[j] = x[j];
      coords.insert(_xx);
      }
  };

  CordFunction cord(u.value_size());
  cord._xx = x;
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  std::vector<double> cell_coefficients(dofmap.max_element_dofs());
  const std::size_t local_size = dofmap.ownership_range().second
                              - dofmap.ownership_range().first;
  for (CellIterator cell(mesh1); !cell.end(); ++cell)
  {
      // Update to current cell
      cell->get_vertex_coordinates(vertex_coordinates);
      cell->get_cell_data(ufc_cell);

      // Call evaluate_dofs with wrapper function to extract points
      element.evaluate_dofs(cell_coefficients.data(), cord,
              vertex_coordinates.data(), ufc_cell.orientation, ufc_cell);
  }
  //   for (auto it = coords.begin(); it != coords.end(); ++it)
  //   {
  //        for (auto y = it->begin(); y != it->end(); ++y)
  //           std::cout << *y << " ";
  //       std::cout << std::endl;
  //   }
  //   std::cout << coords.size() << " " << coords_to_dofs.size() << std::endl;

  // Search this process first for all coordinates in u's local mesh
  std::vector<double> points_not_found;
  for (auto it = coords.begin(); it != coords.end(); ++it)
  {
      // Place interpolation point in x
      std::copy(it->begin(), it->end(), x.begin());

      try
      { // Store values when point is found
      u0.eval(_values, _x);
      coords_to_values.insert(std::make_pair(x, values));
      }
      catch (std::exception &e)
      {
      // If not found then it must be searched on the other processes
      points_not_found.insert(points_not_found.end(), x.begin(), x.end());
      }
  }

  // Get number of MPI processes
  std::size_t num_processes = dolfin::MPI::size(mpi_comm);

  // Remaining interpolation points must be found through MPI
  // communication.  Check first using bounding boxes which process
  // may own the points
  std::vector<std::vector<double>> potential_points(num_processes);
  for (std::size_t i = 0; i < points_not_found.size(); i += gdim1)
  {
      std::copy(points_not_found.begin() + i,
              points_not_found.begin() + i + gdim1, x.begin());

      // Find potential owners
      for (std::size_t p = 0; p < num_processes; p++)
      {
      if (p == dolfin::MPI::rank(mpi_comm))
          continue;

      // Check if in bounding box
      if (in_bounding_box(x, bounding_boxes[p], 1e-12))
      {
          potential_points[p].insert(potential_points[p].end(),
                                  x.begin(), x.end());
      }
      }
  }

  // Communicate all potential points
  std::vector<std::vector<double>> potential_points_recv;
  dolfin::MPI::all_to_all(mpi_comm, potential_points, potential_points_recv);

  // Now try to eval u0 for the received points
  std::vector<std::vector<double>> coefficients_found(num_processes);
  std::vector<std::vector<double>> points_found(num_processes);

  for (std::size_t p = 0; p < num_processes; ++p)
  {
      if (p == dolfin::MPI::rank(mpi_comm))
      continue;

      std::vector<double>& points = potential_points_recv[p];
      for (std::size_t j = 0; j < points.size()/gdim1; ++j)
      {
      std::copy(points.begin() + j*gdim1, points.begin() + (j + 1)*gdim1,
                  x.begin());

      try
      {
          // push back when point is found
          u0.eval(_values, _x);
          coefficients_found[p].insert(coefficients_found[p].end(),
                                      values.begin(), values.end());
          points_found[p].insert(points_found[p].end(), x.begin(), x.end());
      }
      catch (std::exception &e)
      {
          // If not found then do nothing
      }
      }
  }

  // Send back the found coefficients and points
  std::vector<std::vector<double>> coefficients_recv;
  std::vector<std::vector<double>> points_recv;
  dolfin::MPI::all_to_all(mpi_comm, coefficients_found, coefficients_recv);
  dolfin::MPI::all_to_all(mpi_comm, points_found, points_recv);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
      if (p == dolfin::MPI::rank(mpi_comm))
      continue;

      // Get the new values and points
      const std::vector<double>& vals = coefficients_recv[p];
      const std::vector<double>& pts = points_recv[p];

      // Move all found coefficients into the local_u_vector
      for (std::size_t j = 0; j < pts.size()/gdim1; ++j)
      {
      std::copy(pts.begin() + j*gdim1, pts.begin() + (j + 1)*gdim1, x.begin());

      // Store received result in map
      for (uint i = 0; i < u0.value_size(); i++)
          values[i] = vals[j*u0.value_size() + i];
      
      coords_to_values.insert(std::make_pair(x, values));
      }
  }

  // Create an Expression wrapping coords_to_values such that these
  // results will be retrieved when calling evaluate_dofs
  class WrapperFunction : public Expression
  {
  public:

      WrapperFunction(int value_shape) : Expression(value_shape) {};

      mutable std::vector<double> _xx;

      void eval(Array<double>& values, const Array<double>& x) const
      {
      for (uint j = 0; j < x.size(); j++)
              _xx[j] = x[j];

      const std::vector<double>& v = coords_to_values[_xx];
      for (std::size_t j = 0; j < v.size(); j++)
          values[j] = v[j];
      }
  };

  WrapperFunction wrapper(u.value_size());
  wrapper._xx = x;

  // Iterate over mesh and interpolate on each cell
  //   ufc::cell ufc_cell;
  //   std::vector<double> vertex_coordinates;
  //   std::vector<double> cell_coefficients(dofmap.max_element_dofs());
  //
  //   const std::size_t local_size = dofmap.ownership_range().second
  //                                - dofmap.ownership_range().first;
  for (CellIterator cell(mesh1); !cell.end(); ++cell)
  {
      // Update to current cell
      cell->get_vertex_coordinates(vertex_coordinates);
      cell->get_cell_data(ufc_cell);

      // Call evaluate_dofs with wrapper function around the globally
      // computed interpolation points.
      element.evaluate_dofs(cell_coefficients.data(), wrapper,
              vertex_coordinates.data(), ufc_cell.orientation, ufc_cell);

      // Tabulate dofs
      auto cell_dofs = dofmap.cell_dofs(cell->index());

      // Place result in local vector
      for (uint i = 0; i < cell_dofs.size(); i++)
      {
          uint d = cell_dofs[i];
          if (d < local_size)
              local_u_vector[d] = cell_coefficients[i];
      }
  }
  
  coords_to_values.clear();
  coords.clear();
  // Set and finalize vector
  u.vector()->set_local(local_u_vector);
  u.vector()->apply("insert");
}
//
void interpolate_any2(const Expression& u0, Function& u)
{
  // Interpolate from Expression u0 to Function u.
  // This mesh of u0 may be different from that of u
  //

  // Get function spaces of Functions interpolating to
  dolfin_assert(u.function_space());
  const FunctionSpace& V1 =  *u.function_space();

  // Get element interpolating to
  dolfin_assert(V1.element());
  const FiniteElement& element = *V1.element();

  // Check that function ranks match
  if (element.value_rank() != u0.value_rank())
  {
      dolfin_error("interpolate.cpp",
                  "interpolate Expression into function space",
                  "Rank of Expression (%d) does not match rank of function space (%d)",
                  u0.value_rank(), element.value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < element.value_rank(); ++i)
  {
      if (element.value_dimension(i) != u0.value_dimension(i))
      {
      dolfin_error("interpolate.cpp",
                  "interpolate Expression into function space",
                  "Dimension %d of Expression (%d) does not match dimension %d of function space (%d)",
                  i, u0.value_dimension(i), i, element.value_dimension(i));
      }
  }


  // Get mesh and dimension of FunctionSpace interpolating to/from
  dolfin_assert(V1.mesh());
  const Mesh& mesh1 = *V1.mesh();
  const std::size_t gdim1 = mesh1.geometry().dim();

  // Create arrays used to evaluate one point
  std::vector<double> x(gdim1);
  std::vector<double> values(u0.value_size());
  Array<double> _x(gdim1, x.data());
  Array<double> _values(u0.value_size(), values.data());

  // Create vector to hold all local values of u
  std::vector<double> local_u_vector(u.vector()->local_size());

  // Get dofmap of u
  dolfin_assert(V1.dofmap());
  const GenericDofMap& dofmap = *V1.dofmap();

  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  std::vector<double> cell_coefficients(dofmap.max_element_dofs());
  const std::size_t local_size = dofmap.ownership_range().second
                              - dofmap.ownership_range().first;


  // Iterate over mesh and interpolate on each cell
  for (CellIterator cell(mesh1); !cell.end(); ++cell)
  {
      // Update to current cell
      cell->get_vertex_coordinates(vertex_coordinates);
      cell->get_cell_data(ufc_cell);

      // Call evaluate_dofs with wrapper function around the globally
      // computed interpolation points.
      element.evaluate_dofs(cell_coefficients.data(), u0,
              vertex_coordinates.data(), ufc_cell.orientation, ufc_cell);

      // Tabulate dofs
      auto cell_dofs = dofmap.cell_dofs(cell->index());

      // Place result in local vector
      for (uint i = 0; i < cell_dofs.size(); i++)
      {
          uint d = cell_dofs[i];
          if (d < local_size)
              local_u_vector[d] = cell_coefficients[i];
      }
  }

  // Set and finalize vector
  u.vector()->set_local(local_u_vector);
  u.vector()->apply("insert");
}

PYBIND11_MODULE(interpolation, m)
{
  m.def("interpolate", (void (*)(const Function&, Function&))
      &interpolate2);
  m.def("interpolate", (void (*)(const Expression&, Function&))
      &interpolate1);
  m.def("interpolate", [](py::object u0, py::object v){
      auto _u = u0.attr("_cpp_object");
      auto _v = v.attr("_cpp_object").cast<Function*>();
      if (py::isinstance<Function>(_u))
      {
        auto _u0 = _u.cast<const Function*>();
        interpolate2(*_u0, *_v);
      }
      else if (py::isinstance<Expression>(_u))
      {
        auto _u0 = _u.cast<const Expression*>();
        interpolate1(*_u0, *_v);
      }
    });
  m.def("interpolate_any", [](py::object u0, py::object v){
    auto _u = u0.attr("_cpp_object");
    auto _v = v.attr("_cpp_object").cast<Function*>();
    if (py::isinstance<Function>(_u))
    {
      auto _u0 = _u.cast<const Function*>();
      interpolate_any1(*_u0, *_v);
    }
    else if (py::isinstance<Expression>(_u))
    {
      auto _u0 = _u.cast<const Expression*>();
      interpolate_any2(*_u0, *_v);
    }
    });
}
