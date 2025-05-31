import matplotlib as mpl
mpl.use('Agg')
import torch

def linear_HSIC(X, Y):
    n = X.shape[0]
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = (I - unit / n).cuda()
    M = torch.trace(torch.mm(torch.mm(L_X, H), torch.mm(L_Y, H)))
    return M / (n - 1) / (n - 1)


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


def unbias_CKA(X, Y):
    hsic = unbiased_HSIC(X, Y)
    var1 = torch.sqrt(unbiased_HSIC(X, X))
    var2 = torch.sqrt(unbiased_HSIC(Y, Y))
    return hsic / (var1 * var2)


def unbiased_HSIC(X, Y):
    """Unbiased estimator of Hilbert-Schmidt Independence Criterion
    Song, Le, et al. "Feature selection via dependence maximization." 2012.
    """
    kernel_XX = torch.mm(X, X.T)
    kernel_YY = torch.mm(Y, Y.T)

    tK = kernel_XX - torch.diag_embed(torch.diag(kernel_XX))
    tL = kernel_YY - torch.diag_embed(torch.diag(kernel_YY))

    N = kernel_XX.shape[0]
    # print(torch.sum(tK, 0).dot(torch.sum(tL, 1)) / torch.sum(tK @ tL))  # same

    hsic = (
        torch.trace(tK @ tL)
        + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
        - (2 * torch.sum(tK @ tL) / (N - 2))
    )

    return hsic / (N * (N - 3))


# if __name__=='__main__':
#     X = torch.randn((5, 50))
#     Y = torch.randn((5, 10))
#     print(unbiased_HSIC(X, Y))
    # a = torch.Tensor([[1,2],[2,2]])
    # b = torch.Tensor([[2, 1], [1, 3]])
    # print(a, b)
    # print(a @ b)
    # print(torch.mm(a, b))
    # print(torch.sum(a, 0).dot(torch.sum(b, 1)))
    # print(torch.sum(a @ b))
