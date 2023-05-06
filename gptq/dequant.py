import torch

from .quant import QuantLinear


def dequantize(qlinear, bits):
    if bits == 2:
        # Unpack 2bit weights
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qlinear.qweight, 1).expand(-1, 16, -1), qlinear.wf1
        ).to(torch.int8)
        torch.bitwise_and(weight, 0x00000003, out=weight)
        weight = weight.reshape(-1, qlinear.groupsize, weight.shape[2])

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qlinear.qzeros, 2).expand(-1, -1, 16), qlinear.wf2
        ).to(torch.int8)
        torch.bitwise_and(zeros, 0x00000003, out=zeros)
        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = qlinear.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weights = scales * (weight - zeros)
        weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])

    elif bits == 3:
        # Unpack 3bit weights
        weight = qlinear.qweight.reshape(
            qlinear.qweight.shape[0] // 3, 3, 1, qlinear.qweight.shape[1]
        ).expand(-1, -1, 12, -1)
        weight = (weight >> qlinear.wf1) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat(
            [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
        )
        weight = weight.reshape(-1, qlinear.groupsize, weight.shape[2])

        zeros = qlinear.qzeros.reshape(
            qlinear.qzeros.shape[0], qlinear.qzeros.shape[1] // 3, 3, 1
        ).expand(-1, -1, -1, 12)
        zeros = zeros >> qlinear.wf2
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
            (zeros[:, :, 1, 0] << 2) & 0x4
        )
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
            (zeros[:, :, 2, 0] << 1) & 0x6
        )
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2
        )
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
        zeros = zeros + 1

        scales = qlinear.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weights = scales * (weight - zeros)
        weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])

    elif bits == 4:
        # Unpack 4bit weights
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qlinear.qweight, 1).expand(-1, 8, -1), qlinear.wf1
        ).to(torch.int8)
        torch.bitwise_and(weight, 0x0000000F, out=weight)
        weight = weight.reshape(-1, qlinear.groupsize, weight.shape[2])

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qlinear.qzeros, 2).expand(-1, -1, 8), qlinear.wf2
        ).to(torch.int8)
        torch.bitwise_and(zeros, 0x0000000F, out=zeros)
        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = qlinear.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weights = scales * (weight - zeros)
        weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])

    elif bits == 8:
        # Unpack 8bit weights
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qlinear.qweight, 1).expand(-1, 4, -1), qlinear.wf1
        ).to(torch.int8)
        torch.bitwise_and(weight, 0x000000FF, out=weight)
        weight = weight.reshape(-1, qlinear.groupsize, weight.shape[2])

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qlinear.qzeros, 2).expand(-1, -1, 4), qlinear.wf2
        ).to(torch.int8)
        torch.bitwise_and(zeros, 0x000000FF, out=zeros)
        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = qlinear.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weights = scales * (weight - zeros)
        weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    dequantized_linear = torch.nn.Linear(in_features=qlinear.infeatures, out_features=qlinear.outfeatures)

    dequantized_linear.weight = torch.nn.Parameter(weights.transpose(0, 1))
    dequantized_linear.bias = torch.nn.Parameter(qlinear.bias.clone())

    return dequantized_linear


def dequant_layer(module, bits):
    for attr in dir(module):
        tmp = getattr(module, attr)
        if not isinstance(tmp, QuantLinear):
            continue
        delattr(module, attr)
        setattr(
            module,
            attr,
            dequantize(tmp, bits),
        )
    for name1, child in module.named_children():
        dequant_layer(
            child,
            bits,
        )
