import os

os.environ['HF_HOME'] = '/source/changmin/Hunyuan3D-2/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/source/changmin/Hunyuan3D-2/hf_cache'

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/coral1.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/coral1.png')

output_path = 'coral1.glb'
mesh.export(output_path)
print(f"Textured mesh saved to {output_path}")

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/coral2.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/coral2.png')

output_path = 'coral2.glb'
mesh.export(output_path)
print(f"Textured mesh saved to {output_path}")

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/coral3.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/coral3.png')

output_path = 'coral3.glb'
mesh.export(output_path)
print(f"Textured mesh saved to {output_path}")

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/coral5.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/coral5.png')

output_path = 'coral5.glb'
mesh.export(output_path)
print(f"Textured mesh saved to {output_path}")

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/coral6.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/coral6.png')

output_path = 'coral6.glb'
mesh.export(output_path)
print(f"Textured mesh saved to {output_path}")

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/rock1.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/rock1.png')

output_path = 'rock1.glb'
mesh.export(output_path)
print(f"Textured mesh saved to {output_path}")