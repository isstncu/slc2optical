from glob import glob
import os
import numpy as np
import tifffile as tiff

source_dir = r"D:\HuanZhou\S1S2\test"
target_dir = r"D:\HuanZhou\data_crop_256_with_stride_64"
size = (256,256)
stride = (64, 64)

def crop_image(path):
    fileName = os.path.basename(path).replace('.tif','')
    month = fileName.split('_')[0][:6]
    file_source_dir = os.path.dirname(path)
    file_target_dir = file_source_dir.replace(source_dir, target_dir)
    file_target_dir = os.path.join(file_target_dir, month)
    if not os.path.exists(file_target_dir):
        os.makedirs(file_target_dir)
    # 读图
    image_data = tiff.imread(path)
    if len(image_data.shape)==2:
        h, w = image_data.shape
        index = 1
        for height in range(0, h-size[0]+1, stride[0]):
            for width in range(0, w-size[1]+1, stride[1]):
                patch = image_data[height:height+size[0], width:width+size[1]]
                output_path = f'{file_target_dir}/{fileName}_{index}.tif'
                print(f"output path: {output_path}")
                tiff.imwrite(output_path,patch)
                index += 1


    else:
        h, w, _ = image_data.shape
        index = 1
        for height in range(0, h - size[0] + 1, stride[0]):
            for width in range(0, w - size[1] + 1, stride[1]):
                patch = image_data[height:height + size[0], width:width + size[1],:]
                output_path = f'{file_target_dir}/{fileName}_{index}.tif'
                print(f"output path: {output_path}")
                tiff.imwrite(output_path, patch)
                index += 1
                # rows = h // size[0]
    print(path)


if __name__ == '__main__':
    file_list = glob(f"{source_dir}/**/*.tif", recursive=True)
    for file in file_list:
        crop_image(file)

