import os
import glob
from PIL import Image
import numpy as np
import colorsys
import pandas as pd
from sklearn.cluster import KMeans

def extract_colors_kmeans(path, num_colors=3, brightness_thresh=60):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)  

    if arr.shape[2] == 4:
        mask = arr[...,3] > 0
        pix = arr[mask][:,:3]
    else:
        flat = arr[...,:3].reshape(-1,3)
        mask = flat.sum(axis=1) > brightness_thresh
        pix = flat[mask]

    if len(pix) < num_colors:
        raise ValueError(f"{os.path.basename(path)}: usable pixels < num_colors")

    km = KMeans(n_clusters=num_colors, random_state=0).fit(pix)
    centers = km.cluster_centers_.astype(int)
    counts  = np.bincount(km.labels_)

    order = np.argsort(counts)[::-1]
    return centers[order]

def rgb_to_hue(rgb):
    r,g,b = rgb
    h,_,_ = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
    return h

def rank_images(image_dir, num_colors=3, brightness_thresh=60):
    paths = glob.glob(os.path.join(image_dir, "*.png"))
    records = []
    for p in paths:
        try:
            palette = extract_colors_kmeans(p, num_colors, brightness_thresh)
        except ValueError:
            continue
        for rank, rgb in enumerate(palette, 1):
            hue = rgb_to_hue(tuple(rgb))
            records.append({
                "file_name": os.path.basename(p),
                "color_rank": rank,
                "hue": hue
            })

    df = pd.DataFrame(records)
    df = df.sort_values("hue").reset_index(drop=True)
    df.insert(0, "global_rank", np.arange(1, len(df)+1))
    return df

if __name__ == "__main__":
    image_dir = "../data/2d_fishes"
    df = rank_images(image_dir)
    print(df[["global_rank","file_name","color_rank","hue"]])
    df.to_csv("fish_asset_color_ranking_kmeans.csv", index=False)