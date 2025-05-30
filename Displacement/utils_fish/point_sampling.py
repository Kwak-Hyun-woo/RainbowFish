import trimesh
import numpy as np


def fps_sample_vertices(mesh: trimesh.Trimesh,
                        max_samples: int = 100,
                        seed: int = 42) -> np.ndarray:
    """
    Uniformly sample vertices from a mesh using farthest point sampling.
    """
    np.random.seed(seed)
    vertices = np.array(mesh.vertices)
    N = len(vertices)
    
    # pick a random vertex as the first sample
    first_idx = np.random.randint(N)
    selected_idxs = [first_idx]
    
    # 2) calculate the distance from the first vertex to all other vertices
    dist = np.linalg.norm(vertices - vertices[first_idx], axis=1)
    
    # 3) select the next vertex
    for _ in range(1, max_samples):
        # 1) find the vertex with the maximum distance
        next_idx = np.argmax(dist)
        selected_idxs.append(next_idx)
        
        # 2) update the distance to the selected vertex
        new_dist = np.linalg.norm(vertices - vertices[next_idx], axis=1)
        dist = np.minimum(dist, new_dist)
    
    return vertices[selected_idxs]