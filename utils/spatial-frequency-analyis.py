import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像并转为灰度（重要！）
image = plt.imread(r'E:\paper\images\fine_epoch\ALOS_farm_ddpm_output.png')
if len(image.shape) == 3:  # 处理彩色图像
    image = np.mean(image, axis=2).astype(np.uint8)

# 2. 傅里叶变换
fft = np.fft.fft2(image)          # 二维傅里叶变换
fft_shift = np.fft.fftshift(fft)  # 将低频移到频谱中心
magnitude = np.abs(fft_shift)     # 计算幅度谱
log_magnitude = np.log1p(magnitude)  # 对数变换增强可视化

# 3. 绘制结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Spatial Domain'), plt.axis('off')
plt.subplot(122), plt.imshow(log_magnitude, cmap='jet')
plt.title('Frequency Domain (Magnitude)'), plt.axis('off')
plt.tight_layout()
plt.show()