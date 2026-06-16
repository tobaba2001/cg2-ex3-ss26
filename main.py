from pathlib import Path
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh

MESHES = Path("pointcloud")

state = {
    "files": sorted(MESHES.glob("*.off")),
    "selected_index": 0,
    "current_file": None,
    "show_points": True,
    "show_normals": True,
    "pointcloud": None,
    "normal_quantity": None,
}

def load_into_ps(file: Path):
    mesh = trimesh.load_mesh(file, process=False)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    ps.remove_all_structures

    #ps_mesh = ps.register_surface_mesh(
    #    "mesh",
    #    vertices,
    #    faces,
    #    transparency=.4,
    #)
    
    bounding_box_diagonal = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))

    point_radius = .003 * bounding_box_diagonal if state["selected_index"] != 0 else 0.01
    normal_radius = 0.03 * bounding_box_diagonal if state["selected_index"] != 0 else 0.005
    normal_length = 0.03 * bounding_box_diagonal if state["selected_index"] != 0 else 0.05
    
    pointcloud = ps.register_point_cloud(
        "vertices",
        vertices,
        radius= point_radius,
        enabled=state["show_points"],
    )

    normals = np.asarray(mesh.vertex_normals)

    normal_quantity = None

    if normals.shape == vertices.shape:
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(lengths, 1e-12) #TODO

        normal_quantity = pointcloud.add_vector_quantity(
            "vertex normals",
            normals,
            enabled=state["show_normals"],
            length= normal_length,
            radius= normal_radius,
        )


    state["current_file"] = file.name
    state["pointcloud"] = pointcloud
    state["normal_quantity"] = normal_quantity
    

def callback():
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
        if state["selected_index"] == 0:
            ps.set_up_dir('z_up')
            #ps.set_front_dir('x_front')
        else:
            ps.set_up_dir('y_up')

    if state["current_file"] is not None:
        psim.TextUnformatted(f"Loaded: {state['current_file']}")
    
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



ps.init()
ps.set_user_callback(callback)
ps.show()
