import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import open3d as o3d
from open3d.visualization.rendering import Camera

MODEL_PATH    = "/data/final.glb"
VIDEO_DIR     = "/data/outputs/"
Path(VIDEO_DIR).mkdir(parents=True, exist_ok=True)
VIDEO_PATH    = os.path.join(VIDEO_DIR, "rotate_zoom.mp4")
IMAGE_W, IMAGE_H = 1920, 1080

FPS           = 60
DURATION_SEC  = 10
TOTAL_FRAMES  = FPS * DURATION_SEC
START_YAW     = 45.0    
END_YAW       = 90.0    
ZOOM_FACTOR   = 0.2     

SWEEP = 360.0 + (END_YAW - START_YAW)

model    = o3d.io.read_triangle_model(MODEL_PATH)
renderer = o3d.visualization.rendering.OffscreenRenderer(IMAGE_W, IMAGE_H)
scene    = renderer.scene
scene.set_background([0.02,0.02,0.1,1.0])

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
    scene.add_geometry(f"m{i}", mesh_t, mat)

combined = o3d.geometry.TriangleMesh()
for geom in model.meshes:
    combined += geom.mesh
bbox   = combined.get_axis_aligned_bounding_box()
center = bbox.get_center()
radius = np.linalg.norm(bbox.get_extent()) * 1.2

aspect = IMAGE_W / IMAGE_H
v_fov  = 60.0
h_fov  = 2 * np.degrees(np.arctan(np.tan(np.radians(v_fov)/2) * aspect))
scene.camera.set_projection(
    h_fov,
    aspect,
    0.1,
    1000.0,
    Camera.FovType.Horizontal
)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (IMAGE_W, IMAGE_H))

for i in range(TOTAL_FRAMES):
    prog     = i / (TOTAL_FRAMES - 1)
    yaw      = START_YAW + SWEEP * prog
    yaw_rad  = np.deg2rad(yaw)
    elev     = np.deg2rad(15)
    cur_rad  = radius * (1.0 - prog * (1.0 - ZOOM_FACTOR))

    cam_pos = center + cur_rad * np.array([
        np.cos(elev)*np.cos(yaw_rad),
        np.sin(elev),
        np.cos(elev)*np.sin(yaw_rad)
    ])
    scene.camera.look_at(center, cam_pos, [0,1,0])

    img    = renderer.render_to_image()
    pil    = Image.fromarray(np.asarray(img))
    bright = ImageEnhance.Brightness(pil).enhance(1.5)
    frame  = cv2.cvtColor(np.asarray(bright)[:, :, :3], cv2.COLOR_RGB2BGR)

    writer.write(frame)
    if (i+1) % FPS == 0:
        print(f"Encoded {(i+1)//FPS} sec...")

writer.release()
print(f"\n✅ Video saved to {VIDEO_PATH} ({DURATION_SEC} sec @ {FPS} fps)")
