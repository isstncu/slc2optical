import os
from glob import glob
import tifffile as tiff
import numpy as np
import torch
import pandas as pd


real_dir = r"D:\file\result\实验结果\new_data\test_256_with_192_stride_rgb\combine\Australia"
# test_dir = r"D:\file\result\实验结果\CDDPM\new_data\combine\Australia_50_stride"
# test_dir = r"D:\file\result\实验结果\latte5\new_data\combine\M64\sin_cos\Paris_256_with_128_stride\1"
test_dir = r"D:\file\result\实验结果\多任务\Few-U-Dit\combine\decode_alternate\Australia"

def get_files_list(path):
    file_list = os.listdir(path)
    return file_list

def test_rmse(real_file,test_file):
    data1 = tiff.imread(real_file)
    data2 = tiff.imread(test_file)
    data1 = torch.from_numpy(data1).float().permute(2, 0, 1).flatten()
    data2 = torch.from_numpy(data2).float()
    data2 = data2.permute(2,0,1).to(torch.float32).flatten()
    mse = torch.mean((data1 - data2)**2)
    rmse = torch.sqrt(mse)
    return rmse


if __name__ == '__main__':
    df = pd.DataFrame(columns=[str(index) for index in range(1, 13)])
    temp_data={}
    for ch in os.listdir(test_dir):
        file_list = get_files_list(os.path.join(test_dir,ch))
        rmse_list = []
        for file in file_list:
            rmse = test_rmse(real_dir+f"\{file}", os.path.join(test_dir,ch)+f"\{file}")
            rmse_list.append(rmse.item())
        temp_data[ch] = np.mean(rmse_list)
        # print(rmse_list)
        # print(f"average rmse:{np.mean(rmse_list)}")

    df = df.append(temp_data,ignore_index=True)
    df.to_excel("指标1.xlsx",index=False)
    print()