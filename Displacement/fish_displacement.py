import torch
import os
import sys
sys.path.append("./Diffusion-3D-Features")
# os.environ["HF_HOME"] = "./cache"
# os.environ["TRANSFORMERS_CACHE"] = "./cache_transformers"
from time import time
from diff3f import get_features_per_vertex
from utils import convert_mesh_container_to_torch_mesh, cosine_similarity, double_plot, get_colors, generate_colors
from dataloaders.mesh_container import MeshContainer
from diffusion import init_pipe
from dino import init_dino
from functional_map import compute_surface_map

import trimesh
import glob
from trimesh.visual.texture import TextureVisuals, SimpleMaterial
import open3d as o3d
from PIL import Image
import numpy as np
import pyrender
import math
from tqdm import tqdm
import pandas as pd 
import random
import argparse
from utils_fish import fps_sample_vertices, visualize_sorted_vertices_plotly, visualize_direction_vec_w_fish, get_rotation_matrix_from_vectors, rotate_mesh_vertices, rotation_matrix_axis_angle, rotation_matrix_from_vectors, rotate_mesh

def compute_features(device, pipe, dino_model, m, prompt):
    mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex(
        device=device,
        pipe=pipe, 
        dino_model=dino_model,
        mesh=mesh,
        prompt=prompt,
        mesh_vertices=mesh_vertices,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
    )
    return features.cpu()

def align_mesh_rotation_only(reference_mesh, target_mesh, s):
    """
    reference_mesh: reference mesh (trimesh)
    target_mesh: target mesh (trimesh)
    s: torch.Tensor or np.ndarray, shape (N_ref,), reference[i] ↔ target[s[i]]
    """

   # (1) vertex numpy array
    V_ref = np.asarray(reference_mesh.vert)         # (N_ref, 3)
    V_target = np.asarray(target_mesh.vert)         # (N_target, 3)
    V_target_matched = V_target[s]                  # (N_ref, 3)

    # (2) translation ignore
    V_ref_centered = V_ref - V_ref.mean(axis=0)
    V_target_centered = V_target_matched - V_target_matched.mean(axis=0)

    # (3) Covariance Matrix, SVD
    H = V_target_centered.T @ V_ref_centered
    U, S, Vt = np.linalg.svd(H)

    # (4) Make pure rotation matrix
    R = Vt.T @ U.T

    # Reflection caliberation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_align', action='store_false',
                        help='Align meshes before processing')
    parser.add_argument('--num_views', type=int, default=4,
                        help='Number of views per mesh (must be a square number)')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of rendered image')
    parser.add_argument('--width', type=int, default=512,
                        help='Width of rendered image')
    parser.add_argument('--num_images_per_prompt', type=int, default=1,
                        help='Number of images to generate per prompt')
    parser.add_argument('--tolerance', type=float, default=0.004,
                        help='Tolerance value for alignment or matching')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_use_normal_map', action='store_false',
                        help='Whether to use normal maps during rendering')
    parser.add_argument('--data_dir', type=str, default='../data/3d_fishes',
                        help='Directory where mesh data are stored')
    parser.add_argument('--save_dir', type=str, default='../data/school_of_fishes',
                        help='Directory where output data will be stored')
    parser.add_argument('--input_mesh', type=str, default='reference_fish_smpl23_resized.glb',
                        help='Filename of the input mesh')
    parser.add_argument('--num_instances', type=int, default=120,
                        help='Number of mesh instances to render')
    parser.add_argument('--scale_ratio', type=float, default=0.1,
                        help='Scale ratio to resize the mesh')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # ----------------------------------------
    # Basic parameters
    # ----------------------------------------
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)


    ALIGN               = args.no_align
    num_views           = args.num_views
    H                   = args.height
    W                   = args.width
    num_images_per_prompt = args.num_images_per_prompt
    tolerance           = args.tolerance
    random_seed         = args.random_seed
    use_normal_map      = args.no_use_normal_map

    DATA_DIR            = args.data_dir
    SAVE_DIR            = args.save_dir
    INPUT_MESH          = args.input_mesh
    NUM_INSTANCES       = args.num_instances
    SCALE_RATIO         = args.scale_ratio


    print(f"NUM_INSTANCES: {NUM_INSTANCES}, SCALE_RATIO: {SCALE_RATIO}")

    # original mesh load and scale
    reference_fish_path = os.path.join(DATA_DIR, INPUT_MESH)
    reference_fish = trimesh.load(reference_fish_path)
    reference_fish = reference_fish.dump()
    reference_fish = trimesh.util.concatenate(reference_fish)
    
    # Normalize scale of fishes 
    file_path = "fish_asset_color_ranking_kmeans.csv"
    small_fishes = []


    df = pd.read_csv(file_path)
    file_names = [os.path.splitext(name)[0]+"_smpl23_resized.glb" for name in df["file_name"]]
    file_names = [os.path.join(DATA_DIR, name) for name in file_names]
    # file_names = glob.glob(os.path.join(DATA_DIR, "fish*.glb"))

    if NUM_INSTANCES < 200:
        file_names = list(dict.fromkeys(file_names))

    for i, file in enumerate(file_names):
        small_fish = trimesh.load(file)
        small_fish = small_fish.dump()
        small_fish = trimesh.util.concatenate(small_fish)
        small_fishes.append(small_fish)

    points=fps_sample_vertices(reference_fish, max_samples=NUM_INSTANCES)
    sample_indices = sorted(random.sample(range(len(small_fishes)), NUM_INSTANCES))
    small_fishes_sampled = [small_fishes[i] for i in sample_indices]
    small_fishes_sampled_path = [file_names[i] for i in sample_indices]

    if ALIGN == True:
        pipe = init_pipe(device)
        dino_model = init_dino(device)
        rotated_small_fishes = []
        reference_mesh = MeshContainer().load_from_file(reference_fish_path)
        f_reference = compute_features(device, pipe, dino_model, reference_mesh, "fish")
        for small_fish, small_fish_mesh in tqdm(zip(small_fishes_sampled_path, small_fishes_sampled), desc="processing"):
            target_mesh = MeshContainer().load_from_file(small_fish)
            # feature compute
            f_target = compute_features(device, pipe, dino_model, target_mesh, "fish")
            s_sim = cosine_similarity(f_reference.to(device),f_target.to(device))
            s = torch.argmax(s_sim, dim=1).cpu().numpy()
            R = align_mesh_rotation_only(reference_mesh, target_mesh, s)
            T = np.eye(4)
            T[:3, :3] = R
        rotated_small_fishes.append(small_fish_mesh.apply_transform(T))
        processed_fishes = rotated_small_fishes
        align_check = "aligned_"
    else:
        processed_fishes = small_fishes_sampled
        align_check = ""

    # Place small fish instances
    scene = trimesh.Scene()
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    for small_fish, p in tqdm(zip(processed_fishes, sorted_points), desc='Instance Placement'):
        small_fish.apply_scale(SCALE_RATIO)
        small_fish.apply_translation(p)
        scene.add_geometry(small_fish)
    scene.export(os.path.join(SAVE_DIR, f'school_of_different_fishes_{align_check}feat_{NUM_INSTANCES}_test.glb'))