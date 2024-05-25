import json

import torch
import os
import torchmetrics
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
    LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import ToTensor
from tqdm import tqdm

from FrequencySpectrumDistributionSimilarity import FrequencySpectrumDistributionSimilarity

minList = ["LPIPS", "L1InFFT", "L1", "L2InFFT", "L2"]
max_dict = {}
color_table = ["red", "blue"]


class TableRow:
    def __init__(self, paper, overall_scales, overall_metrics, citation=None):
        self.data = {}
        self.paper = paper
        self.citation = citation
        self.overall_scales = overall_scales
        self.overall_metrics = overall_metrics

    def addMetric(self, metric, value, scale):
        if metric not in self.data.keys():
            self.data[metric] = {}
        if scale not in self.data[metric].keys():
            self.data[metric][scale] = []
        self.data[metric][scale].append(value)

    def __str__(self):
        out = f"{self.paper}"
        if self.citation is not None:
            out += f"{self.citation}"

        for metric in self.overall_metrics:
            if metric not in self.data.keys():
                print("ERROR")
                return
            for scale in self.overall_scales:
                if scale in self.data[metric].keys():
                    mean = np.mean(self.data[metric][scale])
                    mark = False
                    index = max_dict[metric][scale].index(mean)
                    # for i in range(len(color_table)):
                    #     if mean == max_dict[metric][scale][i]:
                    #         if mean < 0.01:
                    #             out += f"&\\textcolor{{ {color_table[i]} }} {{ {mean:.3e}({index+1}) }} "
                    #         else:
                    #             out += f"&\\textcolor{{{color_table[i]}}}{{{mean:.3f}({index+1}) }}"
                    #         mark = True
                    # if not mark:
                    #     if mean < 0.01:
                    #         out += f"&{mean:.3e}({index+1}) "
                    #     else:
                    #         out += f"&{mean:.3f}({index+1}) "
                    for i in range(len(color_table)):
                        if mean == max_dict[metric][scale][i]:
                            if metric!="LPIPS" and metric!="SSIM":
                                if mean < 0.01:
                                    out += f"&\\textcolor{{{color_table[i]}}}{{{mean:.2e}}}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"
                                else:
                                    out += f"&\\textcolor{{{color_table[i]}}}{{{mean:.2f}}}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"
                                mark = True
                            else:
                                if mean < 0.01:
                                    out += f"&\\textcolor{{{color_table[i]}}}{{{mean:.3f}}}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"
                                else:
                                    out += f"&\\textcolor{{{color_table[i]}}}{{{mean:.3f}}}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"
                                mark = True
                    if not mark:
                        if metric!="LPIPS" and metric!="SSIM":
                            if mean < 0.01:
                                out += f"&{mean:.2e}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"
                            else:
                                out += f"&{mean:.2f}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"
                        else:
                            if mean < 0.01:
                                out += f"&{mean:.3f}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"
                            else:
                                out += f"&{mean:.3f}\\textcolor{{gray}}{{\\textsuperscript{{{index + 1}}}}}"

                else:
                    out += "&-"
        return out + "\\\\"


def l1_in_fft(gt, img):
    diff = (torch.fft.fftn(gt) - torch.fft.fftn(img)).abs()
    return (diff.sum() / diff.numel()).item()


def l2_in_fft(gt, img):
    diff = (torch.fft.fftn(gt) - torch.fft.fftn(img)).abs()
    return (torch.sum(diff ** 2) / diff.numel()).item()


def psnr_in_fft(gt, img):
    gt_fft = torch.fft.fftn(gt)
    img_fft = torch.fft.fftn(img)
    _psnr = PeakSignalNoiseRatio(data_range=torch.max(gt_fft.abs()).item()).cuda()
    return _psnr(gt_fft.abs(), img_fft.abs()).item()

def psnr(gt, img):
    _psnr = PeakSignalNoiseRatio(data_range=1).cuda()
    return _psnr(gt, img).item()

def l1(gt, img):
    return torch.nn.functional.l1_loss(gt, img).item()

def l2(gt, img):
    return torch.nn.functional.mse_loss(gt, img).item()
convertor = ToTensor()
psnr = PeakSignalNoiseRatio(data_range=1).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
fsds = FrequencySpectrumDistributionSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
l1 = torch.nn.functional.l1_loss
l2 = torch.nn.functional.mse_loss
row_item = {
    "EDSR": {
        "paper": "EDSR",
        "citation": "\citep{edsr}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/edsr_baseline/x2",
            3: "./sr_results/edsr_baseline/x3",
            4: "./sr_results/edsr_baseline/x4"
        },
    },

    "EDSR-LIIF": {
        "paper": "EDSR-LIIF",
        "citation": "\citep{liif}",
        "overall_scales": [2, 3, 4, 6, 12],
        "scale_path": {
            2: "./sr_results/liif/edsr_x2",
            3: "./sr_results/liif/edsr_x3",
            4: "./sr_results/liif/edsr_x4",
            6: "./sr_results/liif/edsr_x6",
            12: "./sr_results/liif/edsr_x12"
        },
    },

    "EDSR-OPESR": {
        "paper": "EDSR-OPESR",
        "citation": "\citep{OPESR}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/OPESR/EDSR_x2",
            3: "./sr_results/OPESR/EDSR_x3",
            4: "./sr_results/OPESR/EDSR_x4"
        },
    },

    "EDSR-SRNO": {
        "paper": "EDSR-SRNO",
        "citation": "\citep{SRNO}",
        "overall_scales": [2, 3, 4, 6, 12],
        "scale_path": {
            2: "./sr_results/SRNO/EDSR_baseline_x2",
            3: "./sr_results/SRNO/EDSR_baseline_x3",
            4: "./sr_results/SRNO/EDSR_baseline_x4",
            6: "./sr_results/SRNO/EDSR_baseline_x6",
            12: "./sr_results/SRNO/EDSR_baseline_x12"
        },
    },

    "EDSR-LTE": {
        "paper": "EDSR-LTE",
        "citation": "\citep{LTE}",
        "overall_scales": [2, 3, 4, ],
        "scale_path": {
            2: "./sr_results/LTE/EDSR_baseline_x2",
            3: "./sr_results/LTE/EDSR_baseline_x3",
            4: "./sr_results/LTE/EDSR_baseline_x4"
        },
    },

    "RDN": {
        "paper": "RDN",
        "citation": "\citep{rdn}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/RDN/RDN_small_x2",
            3: "./sr_results/RDN/RDN_small_x3",
            4: "./sr_results/RDN/RDN_small_x4"
        },
    },

    "RDN-LIIF": {
        "paper": "RDN-LIIF",
        "citation": "\citep{liif}",
        "overall_scales": [2, 3, 4, 6, 12],
        "scale_path": {
            2: "./sr_results/liif/rdn_x2",
            3: "./sr_results/liif/rdn_x3",
            4: "./sr_results/liif/rdn_x4",
            6: "./sr_results/liif/rdn_x6",
            12: "./sr_results/liif/rdn_x12"
        },

    },

    "RDN-OPESR": {
        "paper": "RDN-OPESR",
        "citation": "\citep{OPESR}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/OPESR/RDN_x2",
            3: "./sr_results/OPESR/RDN_x3",
            4: "./sr_results/OPESR/RDN_x4"
        },
    },

    "RDN-LTE": {
        "paper": "RDN-LTE",
        "citation": "\citep{LTE}",
        "overall_scales": [2, 3, 4, 6, 12],
        "scale_path": {
            2: "./sr_results/LTE/RDN_x2",
            3: "./sr_results/LTE/RDN_x3",
            4: "./sr_results/LTE/RDN_x4",
            6: "./sr_results/LTE/RDN_x6",
            12: "./sr_results/LTE/RDN_x12"
        },
    },

    "SwinIR-classical": {
        "paper": "SwinIR-classical",
        "citation": "\citep{swinir}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/SwinIR/swinir_classical_sr_x2",
            3: "./sr_results/SwinIR/swinir_classical_sr_x3",
            4: "./sr_results/SwinIR/swinir_classical_sr_x4"
        },
    },

    "ITSRN": {
        "paper": "ITSRN",
        "citation": "\citep{ITSRN}",
        "overall_scales": [2, 3, 4, 6, 12],
        "scale_path": {
            2: "./sr_results/ITSRN/ITSRN_x2",
            3: "./sr_results/ITSRN/ITSRN_x3",
            4: "./sr_results/ITSRN/ITSRN_x4",
            6: "./sr_results/ITSRN/ITSRN_x6",
            12: "./sr_results/ITSRN/ITSRN_x12"
        },
    },

    "Bicubic": {
        "paper": "Bicubic",
        "citation": "",
        "overall_scales": [2, 3, 4, 6, 12],
        "scale_path": {
            2: "./sr_results/Bicubic/Bicubic_x2",
            3: "./sr_results/Bicubic/Bicubic_x3",
            4: "./sr_results/Bicubic/Bicubic_x4",
            6: "./sr_results/Bicubic/Bicubic_x6",
            12: "./sr_results/Bicubic/Bicubic_x12"
        },
    },

    "HAT-S": {
        "paper": "HAT-S",
        "citation": "\citep{hat}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/HAT/HAT-S_SRx2",
            3: "./sr_results/HAT/HAT-S_SRx3",
            4: "./sr_results/HAT/HAT-S_SRx4"
        },
    },

    "HAT": {
        "paper": "HAT",
        "citation": "\citep{hat}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/HAT/HAT_SRx2",
            3: "./sr_results/HAT/HAT_SRx3",
            4: "./sr_results/HAT/HAT_SRx4"
        },
    },

    "HDSRNet": {
        "paper": "HDSRNet",
        "citation": "\citep{hdsrnet}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/HDSRNet/X2",
            3: "./sr_results/HDSRNet/X3",
            4: "./sr_results/HDSRNet/X4"
        },
    },

    "GRLBase": {
        "paper": "GRLBase",
        "citation": "\citep{grl}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/GRL/base/X2",
            3: "./sr_results/GRL/base/X3",
            4: "./sr_results/GRL/base/X4"
        },
    },

    "GRLSmall": {
        "paper": "GRLSmal",
        "citation": "\citep{grl}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/GRL/small/X2",
            3: "./sr_results/GRL/small/X3",
            4: "./sr_results/GRL/small/X4"
        },
    },

    "GRLTiny": {
        "paper": "GRLTiny",
        "citation": "\citep{grl}",
        "overall_scales": [2, 3, 4],
        "scale_path": {
            2: "./sr_results/GRL/tiny/X2",
            3: "./sr_results/GRL/tiny/X3",
            4: "./sr_results/GRL/tiny/X4"
        },
    }
}


def getGTFileName(name):
    return name[0:4] + ".png"


total_img = 0


def validate_data(dic):
    for key in dic.keys():
        for scale in dic[key]["overall_scales"]:
            path = dic[key]["scale_path"][scale]
            for file in os.listdir(path):
                img = Image.open(os.path.join(path, file))
                gt = Image.open(os.path.join("./sr_results/hr", getGTFileName(file)))
                global total_img
                total_img += 1
                if img.size != gt.size:
                    print(f"Size mismatch for {file}")
                    return False
    return True



with torch.no_grad():
    validate_data(row_item)
    print(f"total:{total_img}")
    print("validate ok")
    pbar = tqdm(total=total_img)
    rows = []
    gts = {}
    for file in os.listdir("./sr_results/hr"):
        gts[file] = convertor(Image.open(os.path.join("./sr_results/hr", file))).cuda()
    for net in row_item.keys():
        row_item[net]["metrics"] = {}
        for scale in row_item[net]["overall_scales"]:
            path = row_item[net]["scale_path"][scale]
            row_item[net]["metrics"][scale] = {}
            for file in os.listdir(path):
                img = convertor(Image.open(os.path.join(path, file))).cuda()
                gt = gts[getGTFileName(file)]
                row_item[net]["metrics"][scale][file] = {"PSNR": psnr(gt.unsqueeze(0), img.unsqueeze(0)).item(),
                                                         "SSIM": ssim(gt.unsqueeze(0), img.unsqueeze(0)).item(),
                                                         "LPIPS": lpips(gt.unsqueeze(0), img.unsqueeze(0)).item(),
                                                         "FSDS": fsds(gt, img),
                                                         "L1InFFT": l1_in_fft(gt, img),
                                                         "L1": l1(gt, img).item(),
                                                         "L2InFFT": l2_in_fft(gt, img),
                                                         "L2": l2(gt, img, reduction="sum").item(),
                                                         "PSNRInFFT": psnr_in_fft(gt, img)}

                pbar.update(1)

with open("div2k_metrics_icml_rebuttal.json", "w") as f:
    json.dump(row_item, f, indent=4)

json_text = ""
with open("div2k_metrics_icml_rebuttal.json", "r") as f:
    json_text = f.read()
dic = json.loads(json_text)
row_item = dic
rows = []
# all_metrics = ["PSNR", "SSIM", "LPIPS", "FSDS", "L1InFFT", "L1", "L2InFFT", "L2", "PSNRInFFT"]
# all_metrics = ["FSDS", "L1InFFT", "L1", "L2InFFT", "L2", "PSNRInFFT"]
# all_metrics = ["FSDS", "PSNRInFFT"]
all_metrics = ["PSNR", "SSIM", "LPIPS", "FSDS"]
for key in row_item.keys():
    row = TableRow(row_item[key]["paper"], [2, 3, 4, 6, 12], all_metrics,None)
    for scale in row_item[key]["overall_scales"]:
        for item in row_item[key]["metrics"][str(scale)]:
            row.addMetric("PSNR", row_item[key]["metrics"][str(scale)][item]["PSNR"], scale)
            row.addMetric("SSIM", row_item[key]["metrics"][str(scale)][item]["SSIM"], scale)
            row.addMetric("LPIPS", row_item[key]["metrics"][str(scale)][item]["LPIPS"], scale)
            row.addMetric("FSDS", row_item[key]["metrics"][str(scale)][item]["FSDS"], scale)
            # row.addMetric("PSNRInFFT", row_item[key]["metrics"][str(scale)][item]["PSNRInFFT"], scale)
            # row.addMetric("PSNR", row_item[key]["metrics"][str(scale)][item]["PSNR"], scale)
            # row.addMetric("L1InFFT", row_item[key]["metrics"][str(scale)][item]["L1InFFT"], scale)
            # row.addMetric("L1", row_item[key]["metrics"][str(scale)][item]["L1"], scale)
            # row.addMetric("L2InFFT", row_item[key]["metrics"][str(scale)][item]["L2InFFT"], scale)
            # row.addMetric("L2", row_item[key]["metrics"][str(scale)][item]["L2"], scale)
            # row.addMetric("PSNRInFFT", row_item[key]["metrics"][str(scale)][item]["PSNRInFFT"], scale)

    rows.append(row)

for metrics in all_metrics:
    max_dict[metrics] = {}
    for scale in [2, 3, 4, 6, 12]:
        if scale not in max_dict[metrics].keys():
            max_dict[metrics][scale] = []
        for row in rows:
            if scale in row.data[metrics].keys():
                max_dict[metrics][scale].append(np.mean(row.data[metrics][scale]))
                if metrics not in minList:
                    max_dict[metrics][scale].sort(reverse=True)
                else:
                    max_dict[metrics][scale].sort()

for row in rows:
    print(row)