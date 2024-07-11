import math
import torch
import einops
import numpy
def frequency_align_integrate(tsr):
    """
    :param tsr: CxHxW
    :return:
    """
    # pad zeros to make H even

    if tsr.shape[1] % 2 == 1:
        tsr = torch.concat([tsr, torch.zeros(tsr.shape[0], 1, tsr.shape[2])], dim=1)

    tsr_rfft = torch.fft.rfft2(tsr, norm="backward")
    part_1 = einops.rearrange(tsr_rfft, "C (p H) W -> C p H W", p=2)
    part_2 = part_1[:, 1, :, :]  # C H/2 W
    part_1 = part_1[:, 0, :, :]  # C H/2 W
    part_2 = torch.flip(part_2, dims=[1]).contiguous()

    part_1 = torch.cumsum(part_1, dim=1)
    part_1 = torch.cumsum(part_1, dim=2)

    part_2 = torch.cumsum(part_2, dim=1)
    part_2 = torch.cumsum(part_2, dim=2)

    return (part_1, part_2)


def _FrequencySpectrumDistributionSimilarity(pred, tar):
    """

    :param pred: the predicted output, in shape CxHxW
    :param tar: the ground-truth data, in shape CxHxW
    :return:
    """
    if pred.shape != tar.shape:
        raise ValueError("The shape of pred is expected to be the same as tar")

    pred = pred - torch.mean(pred)
    pred /= torch.std(pred)
    tar = tar - torch.mean(tar)
    tar /= torch.std(tar)

    pred_part_1, pred_part_2 = frequency_align_integrate(pred)  # C H/2 W
    tar_part_1, tar_part_2 = frequency_align_integrate(tar)  # C H/2 W

    part_1_error = ((pred_part_1 - tar_part_1).abs()) ** 2
    part_2_error = ((pred_part_2 - tar_part_2).abs()) ** 2

    return -10*math.log10(torch.sum(part_1_error + part_2_error) / torch.sum(tar_part_2.abs() ** 2 + tar_part_1.abs() ** 2))


def FrequencySpectrumDistributionSimilarity(pred, gt):
    if pred.shape != gt.shape:
        raise ValueError("The shape of input tensor does not match")
    if len(pred.shape) == 3:
        return _FrequencySpectrumDistributionSimilarity(pred, gt)
    elif len(pred.shape) == 4:
        index = []
        for i in range(pred.shape[0]):
            index.append(_FrequencySpectrumDistributionSimilarity(pred[i], gt[i]))
        return index

def matlab_interface(pred, gt):
    pred = numpy.array(pred)
    gt = numpy.array(gt)
    if len(pred.shape) != 3 or len(gt.shape) != 3:
        print("The input array should be 3D")
        return 0.0
    if pred.shape != gt.shape:
        print("The input array should have the same shape")
        return 0.0
    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)
    pred = einops.rearrange(pred, "H W C -> C H W")
    gt = einops.rearrange(gt, "H W C -> C H W")
    return FrequencySpectrumDistributionSimilarity(pred, gt)
