from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from constraints import ConstraintPoints, compute_constraints
from grid import ImplicitGrid
from point_cloud import PointCloud, load_point_cloud


MESHES = Path(__file__).resolve().parent.parent / "pointcloud"


@dataclass
class PSState:
    files: list[Path]
    selected_idx: int = 0
    point_cloud: PointCloud | None = None
    constraints: ConstraintPoints | None = None
    grid: ImplicitGrid | None = None
    alpha_scale: float = 0.01
    grid_resolution: int = 32
    show_points: bool = True
    show_normals: bool = True
    show_constraints: bool = False

class PSApp:
    state: PSState

    def __init__(self):
        self.state = PSState(files=sorted(MESHES.glob("*.off")))
        self.pointcloud_handle = None
        self.normal_handle = None
        self.constraint_handle = None

    def run(self):
        ps.init()
        ps.set_user_callback(self.callback)
        ps.show()

    def callback(self):
        self._draw_file_loader()
        self._draw_display_controls()
        self._draw_reconstruction_controls()

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

        changed_constraints, self.state.show_constraints = psim.Checkbox(
            "Show constraints",
            self.state.show_constraints,
        )

        if changed_constraints and self.constraint_handle is not None:
            self.constraint_handle.set_enabled(self.state.show_constraints)

    def _draw_reconstruction_controls(self):
        changed_alpha, self.state.alpha_scale = psim.SliderFloat(
            "Alpha scale",
            self.state.alpha_scale,
            0.001,
            0.05,
        )

        if changed_alpha and self.state.point_cloud is not None:
            self.recompute_constraints()

    def load_selected_file(self):
        path = self.state.files[self.state.selected_idx]
        point_cloud = load_point_cloud(path)

        self.state.point_cloud = point_cloud
        self.state.constraints = None
        self.state.grid = None

        ps.remove_all_structures()
        self._register_point_cloud(point_cloud)
        self.recompute_constraints()

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
        )
        self._register_constraints()

    def _register_point_cloud(self, point_cloud: PointCloud):
        point_radius = 0.003 * point_cloud.bbox_diagonal
        normal_radius = 0.03 * point_cloud.bbox_diagonal
        normal_length = 0.03 * point_cloud.bbox_diagonal

        if point_cloud.name == "cat.off":
            point_radius = 0.01
            normal_radius = 0.005
            normal_length = 0.05

        self.pointcloud_handle = ps.register_point_cloud(
            "vertices",
            point_cloud.vertices,
            radius=point_radius,
            enabled=self.state.show_points,
        )

        self.normal_handle = self.pointcloud_handle.add_vector_quantity(
            "vertex normals",
            point_cloud.normals,
            enabled=self.state.show_normals,
            length=normal_length,
            radius=normal_radius,
        )

    def _register_constraints(self):
        if self.state.constraints is None:
            return

        if self.constraint_handle is not None:
            ps.remove_point_cloud("constraints", error_if_absent=False)

        constraints = self.state.constraints
        colors = np.zeros((len(constraints.points), 3))
        colors[constraints.values > 0] = np.array([0.1, 0.35, 1.0])
        colors[constraints.values < 0] = np.array([1.0, 0.25, 0.1])
        colors[constraints.values == 0] = np.array([0.05, 0.05, 0.05])

        radius = 0.004
        if self.state.point_cloud is not None:
            radius *= self.state.point_cloud.bbox_diagonal

        self.constraint_handle = ps.register_point_cloud(
            "constraints",
            constraints.points,
            radius=radius,
            enabled=self.state.show_constraints,
        )
        self.constraint_handle.add_color_quantity(
            "constraint sign",
            colors,
            enabled=True,
        )
