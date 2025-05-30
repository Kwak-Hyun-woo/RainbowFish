import numpy as np
import plotly.graph_objs as go
from matplotlib import cm
import plotly.graph_objects as go
from .point_sampling import fps_sample_vertices


def visualize_sorted_vertices_plotly(vertices):
    vertices = np.array(vertices)
    sorted_indices = np.argsort(vertices[:, 0])
    sorted_vertices = vertices[sorted_indices]

    # Rainbow color mapping
    norm = np.linspace(0, 1, len(sorted_vertices))
    colors = cm.rainbow(norm)
    rgb_colors = (colors[:, :3] * 255).astype(int)  # Convert to RGB

    trace = go.Scatter3d(
        x=sorted_vertices[:, 0],
        y=sorted_vertices[:, 1],
        z=sorted_vertices[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=['rgb({}, {}, {})'.format(r, g, b) for r, g, b in rgb_colors],
        )
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title="Interactive 3D Rainbow Scatter",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )

    fig.show()

def visualize_direction_vec_w_fish(fish_vertices, vector):
    # 벡터 시작점 (예: 원점)
    x0, y0, z0 = 0, 0, 0

    # 벡터 방향 (예: x=1, y=2, z=3)
    u, v, w = vector[0], vector[1], vector[2]
    fish_vertices = fps_sample_vertices(fish_vertices, 100)
    vertices = np.array(fish_vertices)
    sorted_indices = np.argsort(vertices[:, 0])
    sorted_vertices = vertices[sorted_indices]

    # Rainbow color mapping
    norm = np.linspace(0, 1, len(sorted_vertices))
    colors = cm.rainbow(norm)
    rgb_colors = (colors[:, :3] * 255).astype(int)  # Convert to RGB

    trace = go.Scatter3d(
        x=sorted_vertices[:, 0],
        y=sorted_vertices[:, 1],
        z=sorted_vertices[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=['rgb({}, {}, {})'.format(r, g, b) for r, g, b in rgb_colors],
        )
    )

    fig = go.Figure(data=[trace, go.Cone(
        x=[x0], y=[y0], z=[z0],
        u=[u], v=[v], w=[w],
        sizemode="absolute",
        sizeref=0.5,
        anchor="tail",  # 화살표의 tail이 시작점
        showscale=False,
        colorscale='Blues'
    )])
    fig.update_layout(
        title="Interactive 3D Rainbow Scatter",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )

    fig.show()