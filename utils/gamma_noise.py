import numpy
import numpy as np
import torch
import torchvision
from scipy import stats
from sklearn import preprocessing


def gamma_noise(x0, Look=1):
    L = Look
    size = x0.shape
    gamma_noise = np.random.gamma(L, 1/L, size)
    log_gamma_noise = np.log(gamma_noise)
    log_gamma_noise = log_gamma_noise.reshape(-1)
    transformed_data, lambda_ = stats.yeojohnson(log_gamma_noise)
    norm_transformed_data = preprocessing.scale(transformed_data)
    norm_transformed_data = norm_transformed_data.reshape(x0.shape)

    return norm_transformed_data.astype(np.float32)

def gamma_noise_v2(x0,Look=1):
    L = Look
    size = x0.shape
    gamma_noise = np.random.gamma(L, 1/L, size)
    log_gamma_noise = np.log(gamma_noise)
    log_gamma_noise = log_gamma_noise.reshape(-1)
    transformed_data, lambda_ = stats.yeojohnson(log_gamma_noise)
    
    norm_transformed_data = preprocessing.scale(transformed_data)
    norm_transformed_data = norm_transformed_data.reshape(x0.shape)
    # norm_transformed_data = transformed_data.reshape(x0.shape)

    return norm_transformed_data.astype(np.float32), lambda_
#
from scipy.special import inv_boxcox
def inverse_yeojohnson(output, lambda_):
    # 反变换
    #if lambda_ != 0:
        #inverse_data = ((output * lambda_ + 1) ** (1 / lambda_)) - 1
    #else:
        #inverse_data = np.exp(output) - 1
    inverse_data = inv_boxcox(output, lambda_)-1
    return inverse_data

def x0_process(x0):
    img = (x0 * 255).astype(np.uint8)
    img[img == 0] = 1
    img = np.log(img) / np.log(255.0)
    return img.astype(np.float32)

def x0_process_v2(x0, lambda_):
    img = (x0 * 255).astype(np.uint8)
    #img = x0#UC数据集不需要乘255
    img[img == 0] = 1
    log_img = np.log(img)
    log_img = log_img.reshape(-1)
    transformed_data = stats.yeojohnson(log_img, lambda_)
    im_max = stats.yeojohnson(np.log(255), lambda_)
    img = transformed_data.reshape(x0.shape)
    
    img = img / im_max
    return img.astype(np.float32)


def inverse_x0_process(output):
    output = output * np.log(255.0)
    output = np.exp(output).astype(np.uint8)
    return output
    
def inverse_x0_process_v2(output,lambda_):
    im_max = stats.yeojohnson(np.log(255), lambda_)
    print(im_max)
    output = output * im_max
    output = inverse_yeojohnson(output,lambda_)
    output = np.exp(output).astype(np.uint8)
    return output

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def generalized_gamma(x0,theta_0=0.001):
    b = get_beta_schedule(
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    b = torch.from_numpy(b).float()
    t = torch.randint(low=0, high=b.shape[0], size=(x0.size(0) // 2 + 1,))
    a = (1 - b).cumprod(dim=0)  # a: torch.Size([1000])
    k = (b / a) / theta_0 ** 2  # k: torch.Size([1000])
    theta = (a.sqrt() * theta_0).index_select(0, t).view(-1, 1, 1, 1).to(x0.device)  # theta: torch.Size([20, 1, 1, 1])
    k_bar = k.cumsum(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(x0.device)  # k_bar: torch.Size([20, 1, 1, 1])
    a = a.index_select(0, t).view(-1, 1, 1, 1).to(x0.device)  # a: torch.Size([20, 1, 1, 1])
    concentration = torch.ones(x0.size()).to(x0.device) * k_bar  # concentration: torch.Size([20, 1, 128, 128])
    rates = torch.ones(x0.size()).to(x0.device) * theta  # rates: torch.Size([20, 1, 128, 128])
    m = torch.distributions.Gamma(concentration, 1 / rates)  # Gamma object
    e = m.sample().to(x0.device)  # torch.Size([20, 1, 128, 128])
    e = e - concentration * rates
    e = e / (1.0 - a).sqrt()
    return e


def preprocess(img):
    img = numpy.log(img+1)
    img = ((img - numpy.min(img)) / (numpy.max(img) - numpy.min(img))).astype(numpy.float32)
    return img
def postprocess(result):
    result = numpy.exp(result)
    result = (result * 255)
# x0 = np.random.randn(20,1,128,128)
# x = gamma_noise(x0)
# print(x.shape)