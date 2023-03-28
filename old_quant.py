import torch


try:
    import quant_cuda_old
except:
    print("CUDA extension not installed.")


# Assumes layer is perfectly divisible into 256 * 256 blocks
class OldQuantLinear(torch.nn.Module):
    def __init__(self, bits, infeatures, outfeatures):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.bits = bits
        self.register_buffer("zeros", torch.zeros((outfeatures, 1)))
        self.register_buffer("scales", torch.zeros((outfeatures, 1)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 256 * (bits * 8), outfeatures), dtype=torch.int),
        )

    def forward(self, x):
        outshape = list(x.shape)
        x = x.reshape(-1, x.shape[-1])
        y = self.bias.clone().repeat(x.shape[0], 1).float()
        outshape[-1] = self.bias.numel()
        dtype = x.dtype
        x = x.float()
        if self.bits == 2:
            quant_cuda_old.vecquant2matmul(x, self.qweight, y, self.scales.float(), self.zeros.float())
        elif self.bits == 3:
            quant_cuda_old.vecquant3matmul(x, self.qweight, y, self.scales.float(), self.zeros.float())
        elif self.bits == 4:
            quant_cuda_old.vecquant4matmul(x, self.qweight, y, self.scales.float(), self.zeros.float())
        elif self.bits == 8:
            quant_cuda_old.vecquant8matmul(x, self.qweight, y, self.scales.float(), self.zeros.float())
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        y = y.to(dtype)
        return y.reshape(outshape)


def old_make_quant(module, names, bits, groupsize, faster=False, name=""):
    if isinstance(module, OldQuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            setattr(
                module, attr, OldQuantLinear(bits, tmp.in_features, tmp.out_features)
            )
    for name1, child in module.named_children():
        old_make_quant(
            child,
            names,
            bits,
            groupsize,
            faster,
            name + "." + name1 if name != "" else name1,
        )
