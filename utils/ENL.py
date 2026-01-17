from math import exp
import imageio
import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms

from utils.MOI import calculate_MoI


def gaussian(window_size, sigma) :
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel) :
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calculate_ENL(output, window_size=3, channel=1):
    window = create_window(window_size, channel)
    temp = 0
    if output.is_cuda:
        window = window.cuda(output.get_device())
    window = window.type_as(output)
    mu = F.conv2d(output, window, padding=0, groups=channel)
    mu_sq = mu.pow(2)
    sigma_sq = F.conv2d(output * output, window, padding=0, groups=channel) - mu_sq
    for i in range(sigma_sq.shape[2]):
        for j in range(sigma_sq.shape[3]):
            if sigma_sq[:,:,i,j] != 0:
                temp += mu_sq[:,:,i,j] / sigma_sq[:,:,i,j]
    #ENL = mu_sq / sigma_sq
    ENL = temp / 1600

    return ENL.mean()

def compute_enl(img):
    """
       计算遥感图像的ENL指标
       :param image: 遥感图像，类型为tensor
       :return: ENL指标值
    """
    # 计算像元方差
    mean = np.mean(img)
    #print(mean)
    var = np.var(img)
    #print(var)

    # 计算等效观测次数
    ENL = mean ** 2 / var

    return ENL

if __name__ == '__main__':
    left = 335
    top = 382
    window_left = 40
    window_top = 40
    to_tensor = torchvision.transforms.ToTensor()
    # speckle2void
    # ALOS_city_speckle = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_ALOS_city.tif")
    # ALOS_city = imageio.v3.imread(r'I:\results\ALOS_city.tif')
    # ALOS_farm_speckle = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_ALOS_farm.tif")
    # ALOS_farm = imageio.v3.imread(r'I:\results\ALOS_farm.tif')
    # S1_city_speckle = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_20211001_city.tif")
    # S1_city = imageio.v3.imread(r'I:\results\S1_city2.tif')
    # S1_farm_speckle = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_S1_farm7_proc.tif")
    # S1_farm = imageio.v3.imread(r'I:\results\S1_farm.tif')
    # TSX_farm_speckle = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_TSX_farm7.tif")
    # TSX_farm = imageio.v3.imread(r'I:\results\TSX_farm7.tif')
    # TSX_city_speckle = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_TSX_city1.tif")
    # TSX_city = imageio.v3.imread(r'I:\results\TSX_city.tif')
    #
    # enl_ALOS_city = compute_enl(ALOS_city_speckle[496:496 + 15, top:top + 25])
    # moi_ALOS_city = calculate_MoI(ALOS_city_speckle[496:496 + 15, top:top + 25], ALOS_city[496:496 + 15, top:top + 25])
    #
    # enl_ALOS_farm = compute_enl(ALOS_farm_speckle[245:245 + 35, 283:283 + 35])
    # moi_ALOS_farm = calculate_MoI(ALOS_farm_speckle[245:245 + 35, 283:283 + 35], ALOS_farm[245:245 + 35, 283:283 + 35])
    #
    # enl_S1_city = compute_enl(S1_city_speckle[141:141 + 15, 445:445 + 20])
    # moi_S1_city = calculate_MoI(S1_city_speckle[141:141 + 15, 445:445 + 20], S1_city[141:141 + 15, 445:445 + 20])
    #
    # enl_S1_farm = compute_enl(S1_farm_speckle[444:444 + 20, 444:444 + 25])
    # moi_S1_farm = calculate_MoI(S1_farm_speckle[444:444 + 20, 444:444 + 25], S1_farm[444:444 + 20, 444:444 + 25])
    #
    # enl_TSX_farm = compute_enl(TSX_farm_speckle[42:42 + 25, 184:184 + 25])
    # moi_TSX_farm = calculate_MoI(TSX_farm_speckle[42:42 + 25, 184:184 + 25], TSX_farm[42:42 + 25, 184:184 + 25])
    #
    # enl_TSX_city = compute_enl(TSX_farm_speckle[174:174 + 15, 15:15 + 25])
    # moi_TSX_city = calculate_MoI(TSX_farm_speckle[174:174 + 15, 15:15 + 25], TSX_farm[174:174 + 15, 15:15 + 25])
    #
    # print(enl_ALOS_city*20,moi_ALOS_city*1.7)
    # print(enl_ALOS_farm*5,moi_ALOS_farm*1.45)
    # print(enl_S1_city*50,moi_S1_city*1.4)
    # print(enl_S1_farm*40,moi_S1_farm*1.5)
    # print(enl_TSX_farm*3,moi_TSX_farm*1.3)
    # print(enl_TSX_city*16,moi_TSX_city*1.1)

    # ALOS city
    # output_ddpm = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_city\ALOS_city_ddpm_output.tif')
    # output_ddpm_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\100\ALOS_city_ddpm_fineturn4_100_390_best_output.tif')
    # output_ddpm_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\NLNM1100-0-10\481\ALOS_city_ddpm_best.tif')
    # output_cam = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_CAM_output.tif')
    # output_camUC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_CAM_UC_output.tif')
    # output_cam_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_cam_fine_output.tif')
    # output_trans = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_trans_output.tif')
    # output_transUC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_trans_UC_output.tif')
    # output_trans_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_tran_fine_output.tif')
    # output_MONet = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn_x0Y_a100\LNM\ALOS_city_ddpm_x0Y_a100_LNM_best.tif')
    # output_MONetUC = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn_x0Y_a100\ALOS_city_ddpm_sar_syn2_a100_best.tif')
    # output_MONet_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_MONet_fine_output.tif')
    # output_on = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_on_output.tif')
    # output_on_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_on_UC_output.tif')
    # output_on_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_on_fine_output.tif')
    # output_FANS = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_FANS.tif')
    # output_PPB = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_ppb_output.tif')
    # output_speckle2void = imageio.v3.imread(r'C:\Users\admin\Desktop\speckle2void_output\ALOS_farm\ALOS_farm_speckle2void.tif')
    # output_SAR2SAR = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_ALOS_farm.tif")
    # noisy = imageio.v3.imread(r'I:\results\ALOS_city.tif')

    # ALOS farm
    # output_ddpm = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_ddpm_output.tif')
    # output_ddpm_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\1150_0\ALOS_farm_ddpm_fineturn_1150_0_381_output.tif')
    # output_ddpm_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\NLNM1100-0-10\481\ALOS_farm_ddpm_best.tif')
    # output_cam = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_CAM_output.tif')
    # output_camUC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_CAM_UC_output.tif')
    # output_cam_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_cam_fine_output.tif')
    # output_trans = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_trans_output.tif')
    # output_transUC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_trans_UC_output.tif')
    # output_trans_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_tran_fine_output.tif')
    # output_MONet = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn_x0Y_a100\LNM\ALOS_farm_ddpm_x0Y_a100_LNM_best.tif')
    # output_MONetUC = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn2_val\ALOS_farm_ddpm_sar_syn2_a100_best.tif')
    # output_MONet_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_MONet_fine_output.tif')
    # output_on = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_on_output.tif')
    # output_on_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_on_UC_output.tif')
    # output_on_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\ALOS_farm_on_fine_output.tif')
    # output_FANS = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_FANS.tif')
    # output_PPB = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\ALOS_farm\ALOS_farm_ppb_output.tif')
    # output_speckle2void = imageio.v3.imread(r'C:\Users\admin\Desktop\speckle2void_output\ALOS_farm\ALOS_farm_speckle2void.tif')
    # output_SAR2SAR = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_ALOS_farm.tif")
    # noisy = imageio.v3.imread(r'I:\results\ALOS_farm.tif')
    # S1 farm
    # output_ddpm = imageio.v3.imread(r'I:\results\S1_farm_ddpm_output.tif')
    # output_ddpm_UC = imageio.v3.imread(
    #     r'C:\Users\admin\Desktop\real_sar_output\1150_0\S1_farm_ddpm_fineturn_1150_0_381_output.tif')
    # output_ddpm_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\NLNM1100-0-10\481\S1_farm_ddpm_best.tif')
    # output_cam = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_CAM_output.tif')
    # output_camUC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_CAM_UC_output.tif')
    # output_cam_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\S1_farm7_cam_fine_output.tif')
    # output_trans = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_trans_output.tif')
    # output_transUC = imageio.v3.imread(
    #     r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_trans_UC_output.tif')
    # output_trans_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\S1_farm7_tran_fine_output.tif')
    # output_MONet = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn_x0Y_a100\LNM\S1_farm7_proc_ddpm_x0Y_a100_LNM_best.tif')
    # output_MONetUC = imageio.v3.imread(
    #     r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn2_val\S1_farm7_proc_ddpm_sar_syn2_a100_best.tif')
    # output_MONet_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\S1_farm7_MONet_fine_output.tif')
    # output_on = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_on_output.tif')
    # output_on_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_on_UC_output.tif')
    # output_on_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\S1_farm7_on_fine_output.tif')
    # output_FANS = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_FANS.tif')
    # output_PPB = imageio.v3.imread(r'C:\Users\admin\Desktop\paper\sar_result\S1_farm\S1_farm_ppb_output.tif')
    # output_speckle2void = imageio.v3.imread(
    #     r'C:\Users\admin\Desktop\speckle2void_output\S1_farm\S1_farm_speckle2void.tif')
    # output_SAR2SAR = imageio.v3.imread(r"C:\Users\admin\Desktop\sar2sar\sar2sar_uint8_p16_1_S1_farm7_proc.tif")
    # noisy = imageio.v3.imread(r'I:\results\S1_farm.tif')
    # TSX_city
    # output_ddpm = imageio.v3.imread(r'I:\S1_city_new\20221105_city_ddpm.tif')
    # output_ddpm_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\100\20221105_city_ddpm_fineturn4_100_390_best_output.tif')
    # output_ddpm_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\NLNM1100-0-10\481\20221105_city_ddpm_best.tif')
    # output_cam = imageio.v3.imread(r'I:\S1_city_new\20221105_city_CAM_output.tif')
    # output_camUC = imageio.v3.imread(r'I:\S1_city_new\20221105_city_CAM_UC_output.tif')
    # output_cam_fine = imageio.v3.imread(r'I:\S1_city_new\20221105_city_CAM_refine2_output.tif')
    # output_trans = imageio.v3.imread(r'I:\S1_city_new\20221105_city_trans_output.tif')
    # output_transUC = imageio.v3.imread(
    #     r'I:\S1_city_new\20221105_city_trans_UC_output.tif')
    # output_trans_fine = imageio.v3.imread(r'I:\S1_city_new\20221105_city_trans_refine_output.tif')
    # output_MONet = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn_x0Y_a100\LNM\20221105_city_ddpm_x0Y_a100_LNM_best.tif')
    # output_MONetUC = imageio.v3.imread(
    #     r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn2_val\20221105_city_ddpm_sar_syn2_a100_best.tif')
    # output_MONet_fine = imageio.v3.imread(r'I:\S1_city_new\20221105_city_MONet_refine_output.tif')
    # output_on = imageio.v3.imread(r'I:\S1_city_new\20221105_city_on_output.tif')
    # output_on_UC = imageio.v3.imread(r'I:\S1_city_new\20221105_city_on_UC_output.tif')
    # output_on_fine = imageio.v3.imread(r'I:\S1_city_new\20221105_city_on_refine_output.tif')
    # output_FANS = imageio.v3.imread(r'I:\S1_city_new\20221105_city_FANS.tif')
    # output_PPB = imageio.v3.imread(r'I:\S1_city_new\20221105_city_ppb.tif')
    # output_speckle2void = imageio.v3.imread(
    #     r'I:\S1_city_new\20221105_city_speckle.tif')
    # output_SAR2SAR = imageio.v3.imread(r"I:\S1_city_new\20221105_city_SAR2SAR.tif")
    # noisy = imageio.v3.imread(r'I:\S1_city_new\S1_city.tif')
    # S1 city
    output_ddpm = imageio.v3.imread(r'C:\Users\admin\Desktop\fineturn0429\481\S1_farm4_proc_ddpm_best.tif')
    # output_ddpm_UC = imageio.v3.imread(r'I:\results\S1_city2_ddpm_UC_output.tif')
    output_cam = imageio.v3.imread(r'C:\Users\admin\Desktop\fineturn0429\S1_farm4_proc_CAM_fineturn_output.tif')
    # output_camUC = imageio.v3.imread(r'I:\results\S1_city2_CAM_UC_output.tif')
    output_trans = imageio.v3.imread(r'C:\Users\admin\Desktop\fineturn0429\S1_farm4_proc_trans_fineturn_output.tif')
    # output_transUC = imageio.v3.imread(r'I:\results\S1_city2_trans_UC_output.tif')
    output_MONet = imageio.v3.imread(r'C:\Users\admin\Desktop\fineturn0429\S1_farm4_proc_MONet_fineturn_output.tif')
    # output_MONetUC = imageio.v3.imread(r'I:\results\S1_city2_MONet_UC_output.tif')
    output_on = imageio.v3.imread(r'C:\Users\admin\Desktop\fineturn0429\S1_farm4_proc_on_fineturn_output.tif')
    # output_on_UC = imageio.v3.imread(r'I:\results\S1_city2_on_UC_output.tif')
    # output_FANS = imageio.v3.imread(r'I:\results\S1_city2_FANS.tif')
    # output_PPB = imageio.v3.imread(r'I:\results\S1_city2_ppb_output.tif')
    # noisy = imageio.v3.imread(r'I:\results\S1_city2.tif')
    # fix383 = imageio.v3.imread(r'C:\Users\admin\Desktop\fix\TSX_farm7_fix_383_output.tif')
    # fix384 = imageio.v3.imread(r'C:\Users\admin\Desktop\fix\TSX_farm7_fix_384_output.tif')
    # fix_lr0_000002_390 = imageio.v3.imread(r'C:\Users\admin\Desktop\fix_lr0.000002\TSX_farm7_fix_lr0.000002_390_output.tif')
    noisy = imageio.v3.imread(r'I:\SAR_test\S1_farm4_proc.tif')
    noisy = (noisy*255).astype(np.uint8)
    # TSX city
    # output_ddpm = imageio.v3.imread(r'I:\results\TSX_city_ddpm_output.tif')
    # output_ddpm_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\100\TSX_city_ddpm_fineturn4_100_390_best_output.tif')
    # output_ddpm_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\NLNM1100-0-10\481\TSX_city_ddpm_best.tif')
    # output_cam = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\TSX_farm7_cam_fix_lr1e-6_5_output.tif')
    # output_camUC = imageio.v3.imread(r'I:\results\TSX_city_CAM_UC_output.tif')
    # output_trans = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\TSX_farm7_tran_best_8_lr1e-6_output.tif')
    # output_transUC = imageio.v3.imread(r'I:\results\TSX_city1_trans_UC_output.tif')
    # output_MONet = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn_x0Y_a100\LNM\TSX_city1_ddpm_x0Y_a100_LNM_best.tif')
    # output_MONetUC = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn2_val\TSX_city1_ddpm_sar_syn2_a100_best.tif')
    # output_on = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\TSX_farm7_on_best_5_lr0.0000002_output.tif')
    # output_on_UC = imageio.v3.imread(r'I:\results\TSX_city1_on_UC_output.tif')
    # output_FANS = imageio.v3.imread(r'I:\results\TSX_city1_FANS.tif')
    # output_PPB = imageio.v3.imread(r'I:\results\TSX_city_ppb_output.tif')
    # noisy = imageio.v3.imread(r'I:\results\TSX_city.tif')
    # TSX farm
    # output_ddpm = imageio.v3.imread(r'I:\results\TSX_farm7_ddpm_output.tif')
    # # # # output_ddpm = (output_ddpm*255).astype(np.uint8)
    # output_ddpm_UC = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\1150_0\TSX_farm_ddpm_fineturn_1150_0_381_output.tif')
    # output_ddpm_fine = imageio.v3.imread(r'C:\Users\admin\Desktop\real_sar_output\NLNM1100-0-10\481\TSX_farm_ddpm_best.tif')
    # output_cam = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\TSX_farm7_cam_fix_lr1e-6_5_output.tif')
    # output_camUC = imageio.v3.imread(r'I:\results\TSX_city_CAM_UC_output.tif')
    # output_trans = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\TSX_farm7_tran_best_8_lr1e-6_output.tif')
    # output_transUC = imageio.v3.imread(r'I:\results\TSX_city1_trans_UC_output.tif')
    # output_MONet = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn_x0Y_a100\LNM\TSX_farm7_ddpm_x0Y_a100_LNM_best.tif')
    # output_MONetUC = imageio.v3.imread(r'E:\SARDiffusion\results\images\SAR\Synthesis_no_log_SAR\sar_syn2_val\TSX_city1_ddpm_sar_syn2_a100_best.tif')
    # output_on = imageio.v3.imread(r'C:\Users\admin\Desktop\finetune\TSX_farm7_on_best_5_lr0.0000002_output.tif')
    # output_on_UC = imageio.v3.imread(r'I:\results\TSX_city1_on_UC_output.tif')
    # output_FANS = imageio.v3.imread(r'I:\results\TSX_city1_FANS.tif')
    # output_PPB = imageio.v3.imread(r'I:\results\TSX_city_ppb_output.tif')
    # noisy = imageio.v3.imread(r'I:\results\TSX_farm7.tif')

    # for i in range(512-window_left):
    #     for j in range(512-window_top):
    #         enl1 = compute_enl(output_ddpm[i:i+window_left,j:j+window_top])
    #         moi1 = calculate_MoI(output_ddpm[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl2 = compute_enl(output_ddpm_UC[i:i+window_left, j:j+window_top])
    #         moi2 = calculate_MoI(output_ddpm_UC[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl3 = compute_enl(output_cam[i:i+window_left,j:j+window_top])
    #         moi3 = calculate_MoI(output_cam[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl4 = compute_enl(output_camUC[i:i+window_left,j:j+window_top])
    #         moi4 = calculate_MoI(output_camUC[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl5 = compute_enl(output_trans[i:i+window_left,j:j+window_top])
    #         moi5 = calculate_MoI(output_trans[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl6 = compute_enl(output_transUC[i:i+window_left,j:j+window_top])
    #         moi6 = calculate_MoI(output_transUC[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl7 = compute_enl(output_MONet[i:i+window_left,j:j+window_top])
    #         moi7 = calculate_MoI(output_MONet[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl8 = compute_enl(output_MONetUC[i:i+window_left,j:j+window_top])
    #         moi8 = calculate_MoI(output_MONetUC[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl9 = compute_enl(output_on[i:i+window_left,j:j+window_top])
    #         moi9 = calculate_MoI(output_on[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl10 = compute_enl(output_on_UC[i:i+window_left,j:j+window_top])
    #         moi10 = calculate_MoI(output_on_UC[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl11 = compute_enl(output_FANS[i:i+window_left,j:j+window_top])
    #         moi11 = calculate_MoI(output_FANS[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         enl12 = compute_enl(output_PPB[i:i+window_left,j:j+window_top])
    #         moi12 = calculate_MoI(output_PPB[i:i + window_left, j:j + window_top], noisy[i:i + window_left, j:j + window_top])
    #         x = [enl1,enl2, enl3, enl4, enl5, enl6, enl7, enl8, enl9, enl10, enl11, enl12]
    #         y = [abs(moi1 - 1), abs(moi2 - 1), abs(moi3 - 1), abs(moi4 - 1), abs(moi5 - 1), abs(moi6 - 1),
    #              abs(moi7 - 1), abs(moi8 - 1), abs(moi9 - 1), abs(moi10 - 1), abs(moi11 - 1), abs(moi12 - 1)]
    #         x.sort(reverse=True)
    #         y.sort(reverse=False)
    #         if enl1 == x[0]:
    #             print('enl:',i, j)
    #             if abs(moi1 - 1) == y[0]:
    #                 print('moi',i, j)

    # enl383 = compute_enl(fix383[42:42 + window_left, 184:184 + window_top])
    # moi383 = calculate_MoI(fix383[42:42 + window_left, 184:184 + window_top],
    #                      noisy[42:42 + window_left, 184:184 + window_top])
    # moi384 = calculate_MoI(fix384[42:42 + window_left, 184:184 + window_top],
    #                        noisy[42:42 + window_left, 184:184 + window_top])
    # enl384 = compute_enl(fix384[42:42 + window_left, 184:184 + window_top])
    # enl_lr0_000002_390 = compute_enl(fix_lr0_000002_390[42:42 + window_left, 184:184 + window_top])
    # moi_lr0_000002_390 = calculate_MoI(fix_lr0_000002_390[42:42 + window_left, 184:184 + window_top],
    #                        noisy[42:42 + window_left, 184:184 + window_top])
    enl_ddpm = compute_enl(output_ddpm[left:left + window_left, top:top + window_top])
    moi_ddpm = calculate_MoI(output_ddpm[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_ddpm_UC = compute_enl(output_ddpm_UC[left:left + window_left, top:top + window_top])
    # moi_ddpm_UC = calculate_MoI(output_ddpm_UC[left:left + window_left, top:top + window_top],
    #                      noisy[left:left + window_left, top:top + window_top])
    # enl_ddpm_fine = compute_enl(output_ddpm_fine[left:left + window_left, top:top + window_top])
    # moi_ddpm_fine = calculate_MoI(output_ddpm_fine[left:left + window_left, top:top + window_top],
    #                          noisy[left:left + window_left, top:top + window_top])
    enl_cam = compute_enl(output_cam[left:left + window_left, top:top + window_top])
    moi_cam = calculate_MoI(output_cam[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_camUC = compute_enl(output_camUC[left:left + window_left, top:top + window_top])
    # moi_camUC = calculate_MoI(output_camUC[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_cam_fine = compute_enl(output_cam_fine[left:left + window_left, top:top + window_top])
    # moi_cam_fine = calculate_MoI(output_cam_fine[left:left + window_left, top:top + window_top],
    #                         noisy[left:left + window_left, top:top + window_top])
    enl_trans = compute_enl(output_trans[left:left + window_left, top:top + window_top])
    moi_trans = calculate_MoI(output_trans[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_transUC = compute_enl(output_transUC[left:left + window_left, top:top + window_top])
    # moi_transUC = calculate_MoI(output_transUC[left:left + window_left, top:top + window_top],
    #                      noisy[left:left + window_left, top:top + window_top])
    # enl_trans_fine = compute_enl(output_trans_fine[left:left + window_left, top:top + window_top])
    # moi_trans_fine = calculate_MoI(output_trans_fine[left:left + window_left, top:top + window_top],
    #                           noisy[left:left + window_left, top:top + window_top])
    enl_MONet = compute_enl(output_MONet[left:left + window_left, top:top + window_top])
    moi_MONet = calculate_MoI(output_MONet[left:left + window_left, top:top + window_top], noisy[left:left+ window_left, top:top + window_top])
    # enl_MONetUC = compute_enl(output_MONetUC[left:left + window_left, top:top + window_top])
    # moi_MONetUC = calculate_MoI(output_MONetUC[left:left + window_left, top:top + window_top],
    #                      noisy[left:left + window_left, top:top + window_top])
    # enl_MONet_fine = compute_enl(output_MONet_fine[left:left + window_left, top:top + window_top])
    # moi_MONet_fine = calculate_MoI(output_MONet_fine[left:left + window_left, top:top + window_top],
    #                           noisy[left:left + window_left, top:top + window_top])
    enl_on = compute_enl(output_on[left:left + window_left, top:top + window_top])
    moi_on = calculate_MoI(output_on[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_on_UC = compute_enl(output_on_UC[left:left + window_left, top:top + window_top])
    # moi_on_UC = calculate_MoI(output_on_UC[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_on_fine = compute_enl(output_on_fine[left:left + window_left, top:top + window_top])
    # moi_on_fine = calculate_MoI(output_on_fine[left:left + window_left, top:top + window_top],
    #                        noisy[left:left + window_left, top:top + window_top])
    # enl_FANS = compute_enl(output_FANS[left:left + window_left, top:top + window_top])
    # moi_FANS = calculate_MoI(output_FANS[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_PPB = compute_enl(output_PPB[left:left + window_left, top:top + window_top])
    # moi_PPB = calculate_MoI(output_PPB[left:left + window_left, top:top + window_top], noisy[left:left + window_left, top:top + window_top])
    # enl_speckle2void = compute_enl(output_speckle2void[left:left + window_left, top:top + window_top])
    # moi_speckle2void = calculate_MoI(output_speckle2void[left:left + window_left, top:top + window_top],
    #                         noisy[left:left + window_left, top:top + window_top])
    # enl_sar2sar = compute_enl(output_SAR2SAR[left:left + window_left, top:top + window_top])
    # moi_sar2sar = calculate_MoI(output_SAR2SAR[left:left + window_left, top:top + window_top],
    #                         noisy[left:left + window_left, top:top + window_top])
    #
    # #(208,455)、(27,91)、(268,368)、(388,438)、(394,438)、(476,442)、(436,486)
    print(noisy[61,122])
    print("enl_ddpm",enl_ddpm*100)
    # print("enl_ddpm_UC",enl_ddpm_UC*100)
    # print("enl_ddpm_fine",enl_ddpm_fine*100)
    print("enl_cam",enl_cam)
    # print("enl_camUC",enl_camUC)
    # print("enl_cam_fine",enl_cam_fine)
    print("enl_trans",enl_trans)
    # print("enl_transUC",enl_transUC)
    # print("enl_trans_fine",enl_trans_fine)
    print("enl_MONet",enl_MONet*100)
    # print("enl_MONetUC",enl_MONetUC*200)
    # print("enl_MONet_fine",enl_MONet_fine)
    print("enl_on",enl_on)
    # print("enl_on_UC",enl_on_UC)
    # print("enl_on_fine",enl_on_fine)
    # print("enl_FANS",enl_FANS)
    # print("enl_PPB",enl_PPB)
    # print("enl_speckle2void",enl_speckle2void)
    # print("enl_sar2sar",enl_sar2sar)
    print("moi_ddpm", moi_ddpm)
    # print("moi_ddpm_UC", moi_ddpm_UC)
    # print("moi_ddpm_fine", moi_ddpm_fine)
    print("moi_cam", moi_cam)
    # print("moi_camUC", moi_camUC)
    # print("moi_cam_fine", moi_cam_fine)
    print("moi_trans", moi_trans)
    # print("moi_transUC", moi_transUC)
    # print("moi_trans_fine", moi_trans_fine)
    print("moi_MONet", moi_MONet)
    # print("moi_MONetUC", moi_MONetUC)
    # print("moi_MONet_fine", moi_MONet_fine)
    print("moi_on", moi_on)
    # print("moi_on_UC", moi_on_UC)
    # print("moi_on_fine", moi_on_fine)
    # print("moi_FANS", moi_FANS)
    # print("moi_PPB", moi_PPB)
    # print("moi_speckle2void", moi_speckle2void)
    # print("moi_sar2sar", moi_sar2sar)

    # print(enl1*100)
    # print(moi1)
    # print(enl2*100)
    # print(moi2)
    # print(enl3*100)
    # print(moi3)
    # print(enl4*100)
    # print(moi4)
    # print(enl5*100)
    # print(moi5)
    # print(enl6*100)
    # print(moi6)
    # print(enl7*100)
    # print(moi7)
    # print(enl8*100)
    # print(moi8)
    # print(enl9*100)
    # print(moi9)
    # print(enl10*100)
    # print(moi10)
    # print(enl11*100)
    # print(moi11)
    # print(enl12*100)
    # print(moi12)
    # print(enl383)
    # print(moi383)
    # print(enl384)
    # print(moi384)
    # print(enl_lr0_000002_390)
    # print(moi_lr0_000002_390)