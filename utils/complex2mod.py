import torch


def complex2mod(x):
    complex_vh = torch.complex(x[:, 0, :, :], x[:, 1, :, :]).to(torch.complex64)
    complex_vv = torch.complex(x[:, 2, :, :], x[:, 3, :, :]).to(torch.complex64)
    complex_value = torch.stack((complex_vh, complex_vv), dim=1)
    mod_value = torch.abs(complex_value)
    return mod_value

def singleComplex2mod(x):
    complex_value = torch.complex(x[:, 0, :, :], x[:, 1, :, :]).to(torch.complex64)
    # complex_vv = torch.complex(x[:, 2, :, :], x[:, 3, :, :]).to(torch.complex64)
    # complex_value = torch.stack((complex_value), dim=1)
    mod_value = torch.abs(complex_value)
    # print(mod_value.shape)
    return mod_value.unsqueeze(1)

def toComplex(x):
    complex_vh = torch.complex(x[:, 0, :, :], x[:, 1, :, :]).to(torch.complex64)
    complex_vv = torch.complex(x[:, 2, :, :], x[:, 3, :, :]).to(torch.complex64)
    complex_value = torch.stack((complex_vh, complex_vv), dim=1)
    return complex_value

def singleToComplex(x):
    complex_value = torch.complex(x[:, 0, :, :], x[:, 1, :, :]).to(torch.complex64)
    # complex_vv = torch.complex(x[:, 2, :, :], x[:, 3, :, :]).to(torch.complex64)
    # complex_value = torch.stack((complex_vh, complex_vv), dim=1)
    return complex_value.unsqueeze(1)

# if __name__ == '__main__':
#     x = torch.randn((1,4,256,256))
#     print(x)
#     y = torch.fft.fftn(x,dim=(-2, -1))
#     z = torch.fft.ifftn(y,dim=(-2, -1))
#     print(y)
#     print(y.shape)
#     print(z)