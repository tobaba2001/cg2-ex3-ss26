from pathlib import Path
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh
from scipy.spatial import cKDTree

MESHES = Path("../pointcloud")

state = {
    "files": sorted(MESHES.glob("*.off")),
    "selected_index": 0,
    "current_file": None,
    "show_points": True,
    "show_normals": True,
    "pointcloud": None,
    "vertices": None,
    "normal_quantity": None,
    "normals": None,
    "kdtree": None,
    "show_constraints": False,
    "alpha_scale": 0.01,
    "bbox_diagonal": 0,
}

def compute_constraints(points, normals, kdtree, alpha0):

    n = len(points)

    for i in range(n):
        alpha = alpha0
        q = points[i] + alpha * normals[i] # adding the initial vector with a scalar of its normal
        _, neighbor_idx = kdtree.query(q)

        if neighbor_idx == i:
            break;

        alpha *= 0.5


def load_into_ps(file: Path):
    mesh = trimesh.load_mesh(file, process=False)

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    ps.remove_all_structures  
    
    bounding_box_diagonal = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))

    point_radius = 0.003 * bounding_box_diagonal if state["current_file"] != "cat.off" else 0.01
    normal_radius = 0.03 * bounding_box_diagonal if state["current_file"] != "cat.off" else 0.005
    normal_length = 0.03 * bounding_box_diagonal if state["current_file"] != "cat.off" else 0.05
    
    pointcloud = ps.register_point_cloud(
        "vertices",
        vertices,
        radius= point_radius,
        enabled=state["show_points"],
    )

    normal_quantity = None

    if normals.shape == vertices.shape:
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(lengths, 1e-12) #TODO

        normal_quantity = pointcloud.add_vector_quantity(
            "vertex normals",
            normals,
            enabled = state["show_normals"],
            length = normal_length,
            radius = normal_radius,
        )
    
    kdtree = cKDTree(vertices)

    compute_constraints(vertices, normals, 0.01 * bounding_box_diagonal, kdtree)
    
    ps.reset_camera_to_home_view()
    
    state["kdtree"] = kdtree
    state["normals"] = normals
    state["vertices"] = vertices
    state["bbox_diagonal"] = bounding_box_diagonal
    state["current_file"] = file.name
    state["pointcloud"] = pointcloud
    state["normal_quantity"] = normal_quantity
    
def select_off_file():
    psim.TextUnformatted("OFF file loader")

    if len(state["files"]) == 0:
        psim.TextUnformatted(f"No .off files found in {MESHES}")
        return

    file_names = [p.name for p in state["files"]]

    changed, new_idx = psim.Combo(
        "File",
        state["selected_index"],
        file_names,
    )

    if changed:
        state["selected_index"] = new_idx
    
    if psim.Button("Load selected OFF"):
        path = state["files"][state["selected_index"]]
        load_into_ps(path)

    if state["current_file"] is not None:
        psim.TextUnformatted(f"Loaded: {state['current_file']}")
        if state["current_file"] == "cat.off":
            ps.set_up_dir('z_up')
            # ps.set_front_dir('x_front')
        else:
            ps.set_up_dir('y_up')

def toggle_verts_and_normals():
    changed_points, state["show_points"] = psim.Checkbox(
        "Show points",
        state["show_points"],
    )

    if changed_points and state["pointcloud"] is not None:
        state["pointcloud"].set_enabled(state["show_points"])
    
    changed_normals, state["show_normals"] = psim.Checkbox(
        "Show normals",
        state["show_normals"],
    )

    if changed_normals and state["normal_quantity"] is not None:
        state["normal_quantity"].set_enabled(state["show_normals"])


def callback(): 
    select_off_file()
    toggle_verts_and_normals()


ps.init()
ps.set_user_callback(callback)
ps.show()
