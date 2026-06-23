from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from constraints import ConstraintPoints, compute_constraints
from grid import ImplicitGrid, create_grid, evaluate_grid
from point_cloud import PointCloud, load_point_cloud
from marchingCubes import polygonise, GridCell


MESHES = Path(__file__).resolve().parent.parent / "pointcloud"


@dataclass
class PSState:
    files: list[Path]
    selected_idx: int = 0
    point_cloud: PointCloud | None = None
    constraints: ConstraintPoints | None = None
    grid: ImplicitGrid | None = None
    bbox_scale: float = 1.100
    alpha_scale: float = 0.01
    grid_resolution: int = 32
    show_points: bool = True
    show_normals: bool = True
    normals_flipped: bool = False
    show_constraints: bool = False
    show_grid: bool = False
    grid_width: int = 16
    grid_height: int = 16
    grid_depth: int = 16
    radius: float = 0.1
    grid_basis_idx: int = 0
    hide_outsides: bool = False
    show_mesh: bool = False
    iso_level: float = 0.0

class PSApp:
    state: PSState

    def __init__(self):
        self.state = PSState(files=sorted(MESHES.glob("*.off")))
        self.pointcloud_handle = None
        self.normal_handle = None
        self.bbox_handle = None
        self.constraint_handle = None
        self.grid_handle = None
        self.mesh_handle = None

    def run(self):
        ps.init()
        ps.set_user_callback(self.callback)
        ps.show()

    def callback(self):
        self._draw_file_loader()
        self._draw_display_controls()
        self._draw_constraint_controls()
        self._draw_grid_controls()
        self._draw_surface_mesh_controls()

    def _draw_file_loader(self):
        psim.TextUnformatted("OFF file loader")

        if len(self.state.files) == 0:
            psim.TextUnformatted(f"No .off files found in {MESHES}")
            return

        file_names = [path.name for path in self.state.files]
        changed, new_idx = psim.Combo("File", self.state.selected_idx, file_names)

        if changed:
            self.state.selected_idx = new_idx

        if psim.Button("Load selected OFF"):
            self.load_selected_file()

        if self.state.point_cloud is not None:
            psim.TextUnformatted(f"Loaded: {self.state.point_cloud.name}")

    def _draw_display_controls(self):
        changed_points, self.state.show_points = psim.Checkbox(
            "Show points",
            self.state.show_points,
        )

        if changed_points and self.pointcloud_handle is not None:
            self.pointcloud_handle.set_enabled(self.state.show_points)

        changed_normals, self.state.show_normals = psim.Checkbox(
            "Show normals",
            self.state.show_normals,
        )

        if changed_normals and self.normal_handle is not None:
            self.normal_handle.set_enabled(self.state.show_normals)

        if psim.Button("Flip normals") and self.state.point_cloud is not None:
            self.flip_normals()
     
    def _draw_constraint_controls(self):
        psim.TextUnformatted("Constraints")
        psim.Separator()

        changed_constraints, self.state.show_constraints = psim.Checkbox(
            "Show constraints",
            self.state.show_constraints,
        )

        if changed_constraints and self.constraint_handle is not None:
            self.constraint_handle.set_enabled(self.state.show_constraints)

        changed_bbox_scale, self.state.bbox_scale = psim.SliderFloat(
            "Extension Factor",
            self.state.bbox_scale,
            1.0,
            1.5,
        )

        if changed_bbox_scale and self.state.point_cloud is not None:
            self._register_bounding_box(self.state.point_cloud)
            if self.state.constraints is not None:
                self.recompute_constraints()

            if self.state.grid is not None:
                self.recompute_grid()

        changed_alpha, self.state.alpha_scale = psim.SliderFloat(
            "Alpha Factor",
            self.state.alpha_scale,
            0.001,
            0.05,
        )

        if changed_alpha and self.state.point_cloud is not None:
            self.recompute_constraints()

    def _draw_grid_controls(self):
        psim.TextUnformatted("Grid")
        psim.Separator()
        changed_grid, self.state.show_grid = psim.Checkbox(
            "Show grid",
            self.state.show_grid,
        )

        changed_width, self.state.grid_width = psim.SliderInt(
            "# width cells",
            self.state.grid_width,
            2,
            60,
        )

        changed_height, self.state.grid_height = psim.SliderInt(
            "# height cells",
            self.state.grid_height,
            2,
            60,
        )

        changed_depth, self.state.grid_depth = psim.SliderInt(
            "# depth cells",
            self.state.grid_depth,
            2,
            60,
        )
        
        changed_radius, self.state.radius = psim.SliderFloat(
            "Wendland Radius",
            self.state.radius,
            0.01,
            250.0,
        )
        
        basis_labels = ["Constant", "Linear"]
        changed_basis, self.state.grid_basis_idx = psim.Combo(
            "Polynomial Basis",
            self.state.grid_basis_idx,
            basis_labels,
        )

        changed_hide_outsides, self.state.hide_outsides = psim.Checkbox(
            "Hide outside",
            self.state.hide_outsides,
        )

        if changed_grid and self.grid_handle is not None:
            self.grid_handle.set_enabled(self.state.show_grid)

        if (changed_width or changed_height or changed_depth or changed_radius or changed_basis) and self.state.point_cloud is not None:
            self.recompute_grid()
            
        if changed_hide_outsides and self.state.grid is not None:
            self._register_grid()
            
    def _draw_surface_mesh_controls(self):
        psim.TextUnformatted("Surface Mesh")
        psim.Separator()
        
        changed_show_mesh, self.state.show_mesh = psim.Checkbox(
            "Show mesh",
            self.state.show_mesh,
        )
        
        changed_iso_level, self.state.iso_level = psim.SliderFloat(
            "Iso Level",
            self.state.iso_level,
            -1.0,
            1.0,
        )

        # toggle mesh visibility or create/remove mesh
        if changed_show_mesh:
            if self.state.show_mesh:
                # create mesh from current grid
                self._register_mesh()
            else:
                if self.mesh_handle is not None:
                    ps.remove_surface_mesh("implicit_mesh", error_if_absent=False)
                    self.mesh_handle = None

        # if iso level changed, regenerate mesh when visible
        if changed_iso_level and self.state.show_mesh:
            self._register_mesh()

    def load_selected_file(self):
        path = self.state.files[self.state.selected_idx]
        point_cloud = load_point_cloud(path)

        self.state.point_cloud = point_cloud
        self.state.constraints = None
        self.state.grid = None
        self.state.normals_flipped = False

        ps.remove_all_structures()
        self._register_point_cloud(point_cloud)
        self._register_bounding_box(point_cloud)
        self.recompute_constraints()
        self.recompute_grid()

        if point_cloud.name == "cat.off":
            ps.set_up_dir("z_up")
        else:
            ps.set_up_dir("y_up")

        ps.reset_camera_to_home_view()

    def recompute_constraints(self):
        if self.state.point_cloud is None:
            return

        self.state.constraints = compute_constraints(
            self.state.point_cloud,
            self.state.alpha_scale,
            self.state.bbox_scale,
        )
        self._register_constraints()

    def recompute_grid(self):
        if self.state.point_cloud is None:
            return

        bbox_min, bbox_max = self._scaled_bbox(self.state.point_cloud)

        self.state.grid = create_grid(
            bbox_min,
            bbox_max,
            (
                self.state.grid_width,
                self.state.grid_height,
                self.state.grid_depth,
            ),
        )

        if self.state.constraints is not None:
            self.state.grid.values = evaluate_grid(
                self.state.grid.points,
                self.state.constraints,
                self.state.radius,
                basis=["constant", "linear"][self.state.grid_basis_idx],
            )

        self._register_grid()
        # update mesh if visible
        if self.state.show_mesh:
            self._register_mesh()

    def flip_normals(self):
        if self.state.point_cloud is None:
            return

        self.state.point_cloud.normals *= -1
        self.state.normals_flipped = not self.state.normals_flipped

        if self.normal_handle is not None:
            ps.remove_curve_network("vertex normals", error_if_absent=False)

        self._register_normals(self.state.point_cloud)

        if self.state.constraints is not None:
            self.recompute_constraints()

    def _register_point_cloud(self, point_cloud: PointCloud):
        point_radius = 0.003 * point_cloud.bbox_diagonal

        if point_cloud.name == "cat.off":
            point_radius = 0.01

        self.pointcloud_handle = ps.register_point_cloud(
            "vertices",
            point_cloud.vertices,
            radius=point_radius,
            enabled=self.state.show_points,
        )

        self._register_normals(point_cloud)

    def _register_normals(self, point_cloud: PointCloud):
        normal_length = 0.03 * point_cloud.bbox_diagonal
        normal_radius = 0.001 * point_cloud.bbox_diagonal

        if point_cloud.name == "cat.off":
            normal_radius = 0.003

        normal_starts = point_cloud.vertices
        normal_ends = point_cloud.vertices + normal_length * point_cloud.normals

        nodes = np.vstack([normal_starts, normal_ends])
        n_vertices = len(point_cloud.vertices)
        edges = np.column_stack(
            [np.arange(n_vertices), np.arange(n_vertices) + n_vertices]
        )

        self.normal_handle = ps.register_curve_network(
            "vertex normals",
            nodes,
            edges,
            radius=normal_radius,
            enabled=self.state.show_normals,
        )

    def _register_bounding_box(self, point_cloud: PointCloud):
        if self.bbox_handle is not None:
            ps.remove_curve_network("bounding box", error_if_absent=False)

        bbox_min, bbox_max = self._scaled_bbox(point_cloud)
        x_min, y_min, z_min = bbox_min
        x_max, y_max, z_max = bbox_max
        bbox_radius = 0.001 * point_cloud.bbox_diagonal

        if point_cloud.name == "cat.off":
            bbox_radius = 0.001

        nodes = np.array(
            [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ]
        )
        edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
        )

        self.bbox_handle = ps.register_curve_network(
            "bounding box",
            nodes,
            edges,
            radius=bbox_radius,
            enabled=True,
        )

    def _scaled_bbox(self, point_cloud: PointCloud):
        center = 0.5 * (point_cloud.bbox_min + point_cloud.bbox_max)
        half_size = 0.5 * (point_cloud.bbox_max - point_cloud.bbox_min)
        scaled_half_size = self.state.bbox_scale * half_size
        return center - scaled_half_size, center + scaled_half_size

    def _register_constraints(self):
        if self.state.constraints is None:
            return

        if self.constraint_handle is not None:
            ps.remove_point_cloud("constraints", error_if_absent=False)

        constraints = self.state.constraints
        colors = np.zeros((len(constraints.vertices), 3))
        colors[constraints.function_values > 0] = np.array([0.1, 0.35, 1.0])
        colors[constraints.function_values < 0] = np.array([1.0, 0.25, 0.1])
        colors[constraints.function_values == 0] = np.array([0.05, 0.05, 0.05])

        radius = 0.003 * self.state.point_cloud.bbox_diagonal

        if self.state.point_cloud.name == "cat.off":
            radius = 0.01

        self.constraint_handle = ps.register_point_cloud(
            "constraints",
            constraints.vertices,
            radius=radius,
            enabled=self.state.show_constraints,
        )
        self.constraint_handle.add_color_quantity(
            "constraint sign",
            colors,
            enabled=True,
        )

    def _register_grid(self):
        if self.state.grid is None:
            return

        if self.grid_handle is not None:
            ps.remove_point_cloud("grid", error_if_absent=False)

        radius = 0.0015 * self.state.point_cloud.bbox_diagonal
        
        if self.state.point_cloud.name == "cat.off":
            radius = 0.0015

        self.grid_handle = ps.register_point_cloud(
            "grid",
            self.state.grid.points,
            radius=radius,
            enabled=self.state.show_grid,
        )

        if self.state.grid.values is not None:
            colors = np.zeros((len(self.state.grid.points), 3), dtype=float)
            colors[self.state.grid.values > 0] = np.array([1.0, 0.25, 0.1])
            colors[self.state.grid.values < 0] = np.array([0.1, 0.35, 1.0])
            colors[self.state.grid.values == 0] = np.array([0.7, 0.7, 0.7])
            self.grid_handle.add_color_quantity(
                "implicit sign",
                colors,
                enabled=True,
            )
            
        if self.state.hide_outsides and self.state.grid.values is not None:

            mask = self.state.grid.values < 0.0

            filtered_points = self.state.grid.points[mask]
            filtered_values = self.state.grid.values[mask]

            filtered_colors = np.zeros((len(filtered_points), 3))
            filtered_colors[filtered_values < 0] = [0.1, 0.35, 1.0]

            ps.remove_point_cloud("grid", error_if_absent=False)

            self.grid_handle = ps.register_point_cloud(
                "grid",
                filtered_points,
                radius=radius,
                enabled=self.state.show_grid,
            )

            self.grid_handle.add_color_quantity(
                "implicit sign",
                filtered_colors,
                enabled=True,
            )

    def _register_mesh(self):
        """Generate a triangle mesh from the implicit grid using Marching Cubes and register it in Polyscope."""
        if self.state.grid is None or self.state.grid.values is None:
            return

        # remove existing mesh if present
        if self.mesh_handle is not None:
            ps.remove_surface_mesh("implicit_mesh", error_if_absent=False)
            self.mesh_handle = None

        nx, ny, nz = self.state.grid.cell_matrix
        points = self.state.grid.points
        values = self.state.grid.values

        # helper to convert (i,j,k) to flat index in points (meshgrid indexing='ij')
        def idx(i, j, k):
            return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

        verts = []
        faces = []

        # iterate over all cells
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cell = GridCell()
                    # Bourke vertex ordering
                    cell_idx = [
                        (i, j, k),
                        (i + 1, j, k),
                        (i + 1, j + 1, k),
                        (i, j + 1, k),
                        (i, j, k + 1),
                        (i + 1, j, k + 1),
                        (i + 1, j + 1, k + 1),
                        (i, j + 1, k + 1),
                    ]

                    for vi, (ii, jj, kk) in enumerate(cell_idx):
                        fi = idx(ii, jj, kk)
                        cell.p[vi] = points[fi]
                        cell.val[vi] = float(values[fi])

                    try:
                        tris = polygonise(cell, self.state.iso_level)
                    except Exception:
                        tris = []

                    for tri in tris:
                        if tri is None or tri.p[0] is None:
                            continue
                        base = len(verts)
                        verts.append(np.array(tri.p[0], dtype=float))
                        verts.append(np.array(tri.p[1], dtype=float))
                        verts.append(np.array(tri.p[2], dtype=float))
                        faces.append([base, base + 1, base + 2])

        if len(verts) == 0:
            return

        verts = np.vstack(verts)
        faces = np.array(faces, dtype=int)

        self.mesh_handle = ps.register_surface_mesh(
            "implicit_mesh",
            verts,
            faces,
            enabled=self.state.show_mesh,
        )
    