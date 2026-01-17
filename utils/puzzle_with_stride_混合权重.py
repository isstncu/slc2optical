from glob import glob
import os
import numpy as np
import tifffile as tiff

source_dir = r"H:\SAR2光学\S2\2022\croped_2560\test"
target_dir = r"H:\SAR2光学\S2\2022\croped_2560\patches"
target_size = (2560, 2560, 3)
size = (256, 256)
stride = (128, 128)

def puzzle_image(path, weightMatrix):
    # file_list = glob(f"{path}/*.tif")
    file_list = os.listdir(path)
    file_list = [value for value in file_list if value.endswith(".tif")]
    # fileName = file_list[0].split('.')[0].split("_")[0]
    fileName = file_list[0].split('.')[0]
    file_target_dir = target_dir
    if not os.path.exists(file_target_dir):
        os.makedirs(file_target_dir)
    # image_data = tiff.imread(path)
    image_data = np.zeros(target_size)
    count_num = np.zeros(target_size)

    h, w, _ = image_data.shape

    row_list = list(range(0, h - size[0] + 1, stride[0]))
    if row_list[-1] < (h - size[0]):
        row_list.append(h - size[0])
    col_list = list(range(0, w - size[1] + 1, stride[1]))
    if col_list[-1] < (w - size[0]):
        col_list.append(w - size[0])

    index = 1
    for height in row_list:
        for width in col_list:
            file_path = f"{path}/{fileName}_{index}.tif"
            patch = tiff.imread(file_path)
            image_data[height:height + size[0], width:width + size[1], :] += patch*weightMatrix
            count_num[height:height + size[0], width:width + size[1]] += weightMatrix
            index += 1
                # rows = h // size[0]

    image_data = image_data / count_num
    image_data = image_data.astype(np.int16)
    print(f"output dir:{file_target_dir}")
    tiff.imwrite(f"{file_target_dir}/{fileName}.tif", image_data)

def get_dir_list(path):
    folder_list = []
    for root,dirs,files in os.walk(path):
        if "S1" not in root and len(files)>0 and not root.endswith("results"):
            folder_list.append(root)
    return folder_list

def getWeight(blkSize):
    M = 64
    N = (blkSize//2)-M
    # 创建权重向量
    v = np.linspace(1 / M, 1, M, endpoint=True)  # 注意：linspace默认endpoint=True，但我们需要反转数组
    vm = np.tile(v, (M, 1))  # 将v重复M次形成矩阵

    # 构建基本块（注意：这里做了一些调整以适应Python的索引和数组操作）
    # 由于Python的数组索引是从0开始的，我们需要稍微调整以匹配MATLAB的行为
    block_top_left = np.minimum(vm, vm.T)
    block_top_right = np.tile(v, (N, 1)).T

    # block_top_right = block_top_left[:, ::-1]  # 水平翻转
    # block_top_middle = np.tile(v, (N, 1))

    block_bottom_left = np.tile(v, (N, 1))  # 下左角矩阵，垂直翻转的v_extended前半部分
    block_bottom_right = np.ones((N, N))  # 下右角矩阵，全是1

    block = np.concatenate((np.concatenate((block_top_left,block_top_right), axis=1),
                            np.concatenate((block_bottom_left, block_bottom_right), axis=1)), axis=0)

    # 生成最终的权重矩阵
    weight_matrix = np.concatenate((np.concatenate((block, block[:, ::-1]), axis=1),
                                    np.concatenate((block[::-1, :], block[::-1, ::-1]), axis=1)), axis=0)

    return np.stack((weight_matrix,)*3,axis=-1)



if __name__ == '__main__':
    weightMatrix = getWeight(256)
    # weightMatrix = np.ones((256, 256, 3))
    folder_list = get_dir_list(source_dir)
    for folder in folder_list:
        puzzle_image(folder,weightMatrix)
    # puzzle_image(r"D:\HuanZhou\data_crop_64_with_stride_16\test\Australia\S2\202103")
    # file_list = glob(f"{source_dir}/**/*.tif", recursive=True)
    # for file in file_list:
    #     puzzle_image(file)

