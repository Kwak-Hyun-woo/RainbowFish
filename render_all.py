import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import open3d as o3d
from open3d.visualization.rendering import Camera

MODEL_PATH   = "/data/final.glb"
OUTPUT_DIR   = "/data/outputs/"
IMAGE_WIDTH  = 1920
IMAGE_HEIGHT = 1080

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

model    = o3d.io.read_triangle_model(MODEL_PATH)
renderer = o3d.visualization.rendering.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
scene    = renderer.scene

scene.set_background([0.02, 0.02, 0.1, 1.0])

for i, geom in enumerate(model.meshes):
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(geom.mesh)
    mat    = model.materials[i]
    if i == 120:
        mat.shader         = "defaultLitTransparency"
        mat.base_color     = [1,1,1,0.1]
    elif i in [121, 122,123,124,125,126,127]:
        mat.shader         = "defaultLit"
    else:
        mat.shader = "defaultUnlit"
    scene.add_geometry(f"mesh_{i}", mesh_t, mat)

aspect = IMAGE_WIDTH / IMAGE_HEIGHT
v_fov  = 60.0  
h_fov = 2 * np.degrees(np.arctan(np.tan(np.radians(v_fov) / 2) * aspect))
scene.camera.set_projection(
    h_fov,                   
    aspect,                 
    0.1,                     
    1000.0,              
    Camera.FovType.Horizontal
)

combined = o3d.geometry.TriangleMesh()
for geom in model.meshes:
    combined += geom.mesh
bbox   = combined.get_axis_aligned_bounding_box()
center = bbox.get_center()
radius = np.linalg.norm(bbox.get_extent()) * 1.1
angles = np.linspace(0, 360, 8, endpoint=False)

for idx, yaw in enumerate(angles):
    elev    = np.deg2rad(15)
    yaw_rad = np.deg2rad(yaw)
    cam_pos = center + radius * np.array([
        np.cos(elev)*np.cos(yaw_rad),
        np.sin(elev),
        np.cos(elev)*np.sin(yaw_rad)
    ])
    scene.camera.look_at(center, cam_pos, [0,1,0])

    img    = renderer.render_to_image()
    pil    = Image.fromarray(np.asarray(img))
    bright = ImageEnhance.Brightness(pil).enhance(1.5)

    out_path = Path(OUTPUT_DIR) / f"view_{idx:02d}_{int(yaw)}deg.png"
    bright.save(str(out_path))
    print(f"✅ Saved {out_path}")
