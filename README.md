# RainbowFish: rendering mesh of school of fishes
**[KAIST 2025 Spring CS479](https://mhsung.github.io/kaist-cs479-spring-2025/) 3D Rendering Contest**

RainbowFish is a 3D rendering pipeline that reconstructs 2D images into color-sorted 3D fish using Hunyuan 3D-2 and arranges them with Diff3F alignment, achieving a vivid multicolored scene rendered in Open3D.
## Setup

### Generation



#### 2D Fish Images Generation
```
conda env create -n image_gen python=3.8
conda activate image_gen
pip install -r requirements.txt
```
For reference fish, you can use your own fish image and put fish image at `$PROJECT_ROOT$/Generation/`, or you can use reference fish image that we uploaded.

#### 2D Fish Images Background removal
```
pip install --upgrade pip
pip install backgroundremover
```

#### 3D Fish Mesh Reconstruction
Please install requirements via the [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git) site.
\
+`pip install scikit-learn` for Color Sorting
\
+`pip install open3d` for Rendering


### Displacement
#### Mesh Correspondence
```
conda env create -f environment.yaml
conda activate diff3f
```

[Install Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y python=3.10 numpy pandas tqdm pillow matplotlib trimesh open3d pyrender pyqt ipython
pip install pyopengl
```


You might face difficulty in installing pytorch3d or encounter the error `ModuleNotFoundError: No module named pytorch3d` during run time. Unfortunately, this is because pytorch3d could not be installed properly. Please refer [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for alternate ways to install pytorch3d.

## Run
### Generation
#### 2D Fish Images Generation
```
cd $PROJECT_ROOT$/Generation
conda activate image_gen
python generate_2d_fishes.py
```
After generating 2d images of fish with different styles, we manually get rid of images of low quality or with artifacts.
data path: `$PROJECT_ROOT$/data/2d_fishes` 
#### Eliminate Background
```
python rmbackground.py
```

#### 3D Fish & Background Mesh Reconstruction
```
conda activate hunyuan3d
cd $PROJECT_ROOT$/Generation/Hunyuan3D-2
python paint_fishes.py
python paint_background.py
```

#### (Optional) Compress meshes to follow contest guideline
Install `gltf-transform`
```
nvm install 18
nvm use 18
npm install -g @gltf-transform/cli
```
Compress 3D Mesh
(We run 3 times to compress)
```
sh simplify_all.sh
```
Compress Texture map
(We run 1 times to compress)
```
sh resize_all.sh
```

### Displacement

#### Color Sorting
```
conda activate hunyuan3d
cd $PROJECT_ROOT$/Displacement
python compute_HSV.py
```

#### Mesh Correspondence
```
conda activate diff3f
python fish_displacement.py
```

#### Adding Background(e.g. Coral, Rock)

We merge all meshes including school of fishes and background assets by using `Blender`

### Rendering 
If you want to render our final result without any procedure, then load [this file](https://drive.google.com/file/d/1A7hphlVOOMHZ908SQGgQhv72lVoyUFkZ/view?usp=sharing), unzip this file and place `final.glb`, and `final_2.glb` at `$PROJECT_ROOT$/data/`. First, you need to try with `final.glb`. if you have some problem about compatibility, then you can try with `final_2.glb`.
```
cd $PROJECT_ROOT$
conda activate hunyuan3d 
```
#### Images
```
python render_all.py
```
#### Video
```
python render_video.py
```
Render Image and video save directory: `$PROJECT_ROOT$/data`
## References
Code reference about Stable Diffusion\
https://huggingface.co/learn/cookbook/en/stable_diffusion_interpolation


Background remover\
https://github.com/nadermx/backgroundremover

Diff3F\
https://github.com/niladridutt/Diffusion-3D-Features 

Hunyuan3D-2\
https://github.com/Tencent-Hunyuan/Hunyuan3D-2