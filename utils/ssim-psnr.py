import imageio
import numpy
import numpy as np
import torchvision
from metrics import calculate_ssim,calculate_psnr

if __name__ == '__main__':
    to_tensor = torchvision.transforms.ToTensor()
    # namedist = ['20220630_farm_41','20220630_farm_47','20220630_farm_49','20220630_farm_52','20220630_farm_71','20220630_farm_75','20220630_farm_95','20220630_farm_102','20220630_farm_116','20220630_farm_125']
    # namedist = ['20220813_port_433','20220813_port_435','20220813_port_436','20220813_port_443','20220813_port_444','20220813_port_460','20230314_port_486','20230314_port_503','20230428_port_528','20230428_port_536']
    # namedist = ['20220630_forest_18','20220630_forest_24','20220630_forest_37','20220630_forest_40','20220630_forest_52','20220630_forest_63','20220630_forest_72','20220630_forest_82','20220630_forest_83','20220630_forest_91']
    # namedist = ['20220813_city_26','20220813_city_32','20220813_city_34','20230314_city_67','20230314_city_68','20230314_city_79','20230314_city_81','20230314_city_94','20230314_city_108','20230314_city_110']
    namedist = ['20220620_farm_forest_360_122','20220620_farm_forest_360_209','20220620_farm_forest_360_216','20220630_farm_forest_800_51','20220630_farm_forest_800_59','20220630_farm_forest_800_81','20220630_farm_forest_800_82','20220630_farm_forest_800_133','20220630_farm_forest_800_153','20220630_farm_forest_800_154','20220630_farm_forest_800_188','20220630_farm_forest_800_583','20220630_farm_forest_800_634','20220630_farm_forest_800_752','20220813_city_224_96','20220813_port_196_193','20230314_city_616_50','20230314_city_616_110','20230314_city_616_126','20230428_port_150_76','20230428_port_400_223','20230428_port_400_281','20230503_city_30_28','20230517_forest_rgb_280_227','20230525_port_100_3']
    ssim1 = ssim2 = ssim3 = ssim4 = ssim5 = ssim6 = ssim7 = 0
    psnr1 = psnr2 = psnr3 = psnr4 = psnr5 = psnr6 = psnr7 = 0
    std_clean = std0 = std1 = std2 = std3 = std4 = std5 = std6 = std7 = 0
    for name in namedist:
        output_ddpm = imageio.v3.imread(fr'C:\Users\admin\Desktop\synthesis\0701\step=30w\sample\{name}.tif')
        # output_cam = imageio.v3.imread(fr'C:\Users\admin\Desktop\4L-output\{name}_CAM_L4_output.tif')
        # # output_camUC = imageio.v3.imread(r'E:\SAR-CAM\data\area_test\20211001_farm_UC_output.tif')
        # output_trans = imageio.v3.imread(fr'C:\Users\admin\Desktop\4L-output\{name}_tran_L4_output.tif')
        # # output_transUC = imageio.v3.imread(r'E:\sar_transformer\area_test\20211001_farm_UC_output.tif')
        # output_MONet = imageio.v3.imread(fr'C:\Users\admin\Desktop\4L-output\{name}_MONet_L4_output.tif')
        # output_on = imageio.v3.imread(fr'C:\Users\admin\Desktop\4L-output\{name}_on_L4_output.tif')
        # output_PPB = imageio.v3.imread(fr'C:\Users\admin\Desktop\4L-output\{name}_ppb_L4.tif')
        # output_FANS = imageio.v3.imread(fr'C:\Users\admin\Desktop\4L-output\{name}_FANS_L4.tif')
        clean = imageio.v3.imread(fr'C:\Users\admin\Desktop\synthesis\无mse\gt\{name}.tif')
        # clean = (clean*255).astype(numpy.uint8)
        # noisy = imageio.v3.imread(fr'E:\SARDiffusion\scratch\data\synthesis_v2\syn_ture\L1\{name}.tif')
        # noisy = (noisy*255).astype(numpy.uint8)
        std_clean += np.std(clean)
        # std0 += np.std(noisy)

        ssim1 += calculate_ssim(to_tensor(output_ddpm).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        psnr1 += calculate_psnr(to_tensor(output_ddpm).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        std1 += np.std(output_ddpm)

        # ssim2 += calculate_ssim(to_tensor(output_cam).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # psnr2 += calculate_psnr(to_tensor(output_cam).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # std2 += np.std(output_cam)
        #
        # ssim3 += calculate_ssim(to_tensor(output_trans).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # psnr3 += calculate_psnr(to_tensor(output_trans).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # std3 += np.std(output_trans)
        #
        # ssim4 += calculate_ssim(to_tensor(output_MONet).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # psnr4 += calculate_psnr(to_tensor(output_MONet).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # std4 += np.std(output_MONet)
        #
        # ssim5 += calculate_ssim(to_tensor(output_on).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # psnr5 += calculate_psnr(to_tensor(output_on).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # std5 += np.std(output_on)
        #
        # ssim6 += calculate_ssim(to_tensor(output_PPB).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # psnr6 += calculate_psnr(to_tensor(output_PPB).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # std6 += np.std(output_PPB)
        #
        # ssim7 += calculate_ssim(to_tensor(output_FANS).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # psnr7 += calculate_psnr(to_tensor(output_FANS).unsqueeze(0), to_tensor(clean).unsqueeze(0))
        # std7 += np.std(output_FANS)


    print(std_clean / 10)
    print(std0 / 10)
    print(ssim1 / 25, psnr1 / 25, std1 / 10)
    print(ssim2 / 10, psnr2 / 10, std2 / 10)
    print(ssim3 / 10, psnr3 / 10, std3 / 10)
    print(ssim4 / 10, psnr4 / 10, std4 / 10)
    print(ssim5 / 10, psnr5 / 10, std5 / 10)
    print(ssim6 / 10, psnr6 / 10, std6 / 10)
    print(ssim7 / 10, psnr7 / 10, std7 / 10)
