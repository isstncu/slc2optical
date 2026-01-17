import imageio
import numpy as np
import warnings

def calculate_EPD_ROA(img, denoised_img):
    # 计算原始图像和去噪图像的HD水平方向和垂直VD方向的EPD-ROA
    I1, I2 = 0, 0
    I3, I4 = 0, 0
    for j in range(img.shape[1]):
        for i in range(img.shape[0]-1):
            temp1 = np.abs(np.divide(denoised_img[i][j], denoised_img[i+1][j]))
            if temp1 <= 10:
                I1 = temp1 + I1
            temp2 = np.abs(img[i][j] / img[i + 1][j])
            if temp2 <= 10:
                I2 = temp2 +I2
    VD = I1 / I2
    for x in range(img.shape[0]):
        for y in range(img.shape[1]-1):
            temp3 = np.abs(denoised_img[x][y] / denoised_img[x][y+1])
            if temp3 <= 10:
                I3 = temp3 + I3
            temp4 = np.abs(img[x][y] / img[x][y+1])
            if temp4 <= 10:
                I4 = temp4 + I4
    HD = I3 / I4
    return HD, VD

warnings.filterwarnings("ignore")
# output_v7 = imageio.v3.imread(r'C:\Users\admin\Desktop\S2train\20211001_farm_ddpm_output.jpg')
# output_cam = imageio.v3.imread(r'C:\Users\admin\Desktop\S2train\20211001_farm_output_cam.jpg')
# # output_camUC = imageio.v3.imread(r'E:\SAR-CAM\data\area_test\20211001_farm_UC_output.tif')
# output_trans = imageio.v3.imread(r'C:\Users\admin\Desktop\S2train\20211001_farm_output_trans.jpg')
# # output_transUC = imageio.v3.imread(r'E:\sar_transformer\area_test\20211001_farm_UC_output.tif')
# output_MONet = imageio.v3.imread(r'C:\Users\admin\Desktop\S2train\20211001_farm_output_MONet.jpg')
# output_PPB = imageio.v3.imread(r'C:\Users\admin\Desktop\S2train\20211001_farm_output_PPB.tif')
# noisy = imageio.v3.imread(r'C:\Users\admin\Desktop\S2train\20211001_farm_origin.jpg')

output_v7 = imageio.v3.imread('../results/images/SAR/Synthesis_no_log_SAR/v22/20211001_farm_output.tif')
output_cam = imageio.v3.imread(r'E:\SAR-CAM\data\area_test\20211001_farm_output.tif')
output_camUC = imageio.v3.imread(r'E:\SAR-CAM\data\area_test\20211001_farm_UC_output.tif')
output_trans = imageio.v3.imread(r'E:\sar_transformer\area_test\20211001_farm_output.tif')
output_transUC = imageio.v3.imread(r'E:\sar_transformer\area_test\20211001_farm_UC_output.tif')
output_MONet = imageio.v3.imread(r'E:\MONet\area_test\20211001_farm_output1.tif')
output_PPB = imageio.v3.imread(r'H:\ppbNakagami\20211001_farm_output.tif')
noisy = imageio.v3.imread('../scratch/data/synthesis_v1/area_test/20211001_farm.tif')

# for i in range(128, 256):
#     for j in range(128, 256):
#         print(i,j)
#         result11,result12 = calculate_EPD_ROA(output_v7[i:i+256, j:j+256], noisy[i:i+256, j:j+256])
#         result21,result22 = calculate_EPD_ROA(output_cam[i:i+256, j:j+256], noisy[i:i+256, j:j+256])
#         result31,result32 = calculate_EPD_ROA(output_camUC[i:i+256, j:j+256], noisy[i:i+256, j:j+256])
#         result41,result42 = calculate_EPD_ROA(output_trans[i:i+256, j:j+256], noisy[i:i+256, j:j+256])
#         result51,result52 = calculate_EPD_ROA(output_transUC[i:i+256, j:j+256], noisy[i:i+256, j:j+256])
#         if result11 == max(result11,result21,result31,result41,result51) and result12 == max(result12,result22,result32,result42,result52) and result11<1 and result12 <1:
#             print("results:",i,j)
result11, result12 = calculate_EPD_ROA(noisy, output_v7)
# result21, result22 = calculate_EPD_ROA(output_v7UC, noisy)
result31, result32 = calculate_EPD_ROA(noisy, output_cam)
#result41, result42 = calculate_EPD_ROA(noisy[256:512,256:512], output_camUC[256:512,256:512])
result51, result52 = calculate_EPD_ROA(noisy, output_trans)
#result61, result62 = calculate_EPD_ROA(noisy[256:512,256:512], output_transUC[256:512,256:512])
result71, result72 = calculate_EPD_ROA(noisy, output_MONet)
result81, result82 = calculate_EPD_ROA(noisy, output_PPB)
print(result11, result12)
#print(result21, result22)
print(result31, result32)
#print(result41, result42)
print(result51, result52)
#print(result61, result62)
print(result71, result72)
print(result81, result82)
