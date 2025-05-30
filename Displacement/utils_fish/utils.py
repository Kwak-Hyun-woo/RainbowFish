# Direction modification
import numpy as np
from .point_sampling import fps_sample_vertices

def get_main_axis(mesh):
    reduced_vertices = fps_sample_vertices(mesh, max_samples=100)
    centered = reduced_vertices - reduced_vertices.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    axis = Vt[0]  # 가장 큰 주성분 → 길이 방향
    return axis  # 방향은 정해지지 않음 (머리인지 꼬리인지 아직 모름)

def project_along_axis(mesh, axis):
    proj = mesh.vertices @ axis
    return proj  # 각 vertex가 axis 위에 어느 위치에 있는지를 나타냄

def detect_head_direction(mesh, axis, ratio=0.05):
    proj = project_along_axis(mesh, axis)
    min_proj, max_proj = np.min(proj), np.max(proj)

    # 양 끝의 vertex subset
    head_side = mesh.vertices[(proj > max_proj - (max_proj - min_proj) * ratio)]
    tail_side = mesh.vertices[(proj < min_proj + (max_proj - min_proj) * ratio)]

    # 단순히 각 side의 vertex 분산으로 비교
    head_var = np.var(head_side @ axis)
    tail_var = np.var(tail_side @ axis)

    # 더 넓은 쪽을 head로 판단
    if head_var > tail_var:
        return axis  # 현재 axis 방향이 head→tail
    else:
        return -axis  # 반대로 뒤집음
    
def get_fish_forward_vector(mesh):
    axis = get_main_axis(mesh)
    "Done calculating axis!!"
    direction = detect_head_direction(mesh, axis)
    "Done calculating direction!!"
    return direction  # 방향 벡터