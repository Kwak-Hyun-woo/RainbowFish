from scipy.spatial.transform import Rotation as R
import numpy as np
import trimesh

def rotation_matrix_from_vectors(vec1, vec2):
    rotation, _ = R.align_vectors([vec2], [vec1])  # vec1 → vec2로 맞춤
    return rotation.as_matrix()

def rotate_mesh(mesh, R_mat):
    mesh.vertices = np.dot(mesh.vertices, R_mat.T)
    return mesh
def rotation_matrix_axis_angle(axis, angle):
    """
    회전 축(axis)과 회전 각도(angle)를 받아 3x3 회전 행렬을 반환합니다.
    axis: (3,) numpy array - 단위 벡터로 정규화된 회전 축
    angle: float - 라디안 단위의 회전 각도
    """
    axis = axis / np.linalg.norm(axis)  # 단위 벡터화
    x, y, z = axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1.0 - cos_theta

    R = np.array([
        [cos_theta + x * x * one_minus_cos,
         x * y * one_minus_cos - z * sin_theta,
         x * z * one_minus_cos + y * sin_theta],

        [y * x * one_minus_cos + z * sin_theta,
         cos_theta + y * y * one_minus_cos,
         y * z * one_minus_cos - x * sin_theta],

        [z * x * one_minus_cos - y * sin_theta,
         z * y * one_minus_cos + x * sin_theta,
         cos_theta + z * z * one_minus_cos]
    ])
    return R

def get_rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, 1.0):  # 이미 정렬됨
        return np.eye(3)
    elif np.isclose(c, -1.0):  # 정반대 방향이면 특수 처리 필요
        # 벡터 a와 직교하는 벡터 하나를 임의로 선택
        ortho = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        v = np.cross(a, ortho)
        v = v / np.linalg.norm(v)
        return rotation_matrix_axis_angle(v, np.pi)
    s = np.linalg.norm(v)
    kmat = np.array([[  0, -v[2],  v[1]],
                     [ v[2],   0, -v[0]],
                     [-v[1], v[0],   0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return R
def rotate_mesh_vertices(mesh, R_mat):
    mesh.vertices = mesh.vertices @ R_mat.T  # 회전 적용
    return mesh