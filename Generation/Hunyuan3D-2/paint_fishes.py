import os
from pathlib import Path
import torch, gc

os.environ['HF_HOME'] = './Hunyuan3D-2/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = './Hunyuan3D-2/hf_cache'

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
paint_pipe = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

image_dir   = Path('../../data/2d_fishes')            
output_dir  = Path('../../data/3d_fishes')         
output_dir.mkdir(parents=True, exist_ok=True)

valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}

for img_path in sorted(image_dir.iterdir()):
    if img_path.suffix.lower() not in valid_exts:
        continue

    print(f'▶ {img_path.name} Processing …')
    mesh = shape_pipe(image=str(img_path))[0]
    mesh = paint_pipe(mesh, image=str(img_path))
    out_path = output_dir / f'{img_path.stem}.glb'
    mesh.export(out_path)
    print(f'   ↳ Save Complete: {out_path}')

    # 4) 메모리 정리
    del mesh
    torch.cuda.empty_cache()
    gc.collect()

print('✅ Finished!')