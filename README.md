# Exploring the Low-Pass Filtering Behavior in Image Super-Resolution

Haoyu Deng, Zijing Xu, Yule Duan, Xiao Wu, Wenjie Shu, Liang-Jian Deng<sup>†</sup>
<sup>†</sup>Corresponding author


If you have any questions, feel free to raise an issue or send a mail to academic@hydeng.cn. I will respond to you as soon as possible. If you think our work is useful, please give us a warmful citation:
```
@article{deng2024exploring,
  title={Exploring the Low-Pass Filtering Behavior in Image Super-Resolution},
  author={Deng, Haoyu and Xu, Zijing and Duan, Yule and Wu, Xiao and Shu, Wenjie and Deng, Liang-Jian},
  journal={arXiv preprint arXiv:2405.07919},
  year={2024}
}
```

## TODOs
- [ ] matlab implementation of FSDS
- [ ] Try to use FSDS as a loss and share results

## Impulse Responses
Please refer to `Impulse_Responses.pptx`. We provide two versions, with/without enhancement. We use the enhancement provided by PPT for better visualization. The enhanced figures are brighter in color while the unenhanced ones are mathematically closer to the sinc funtion. If your directly observe the output of networks (before clamp), you will find another feature of the sinc function: negative values near the main lobe.

## Hybrid Response Analysis (HyRA)
We provide a script to directly analyze a network using HyRA, i.e., `HyRA.py`. Here is the usage

```bash
python HyRA.py --lr [path to low-resolution image] --sr [path to super-resolution image, namely N(I) in the paper] --scale [the scale of super resolution] --impulse_response [path to impulse response] --save_path [path to save results]
```

We also provide a tutorial for the code. Please refer to `HyRA Usage.ipynb`.

## Frequency Spectrum Distribution Similarity (FSDS)
FSDS describes the image quality from the perspective of frequency spectrum. The complete implementation of FSDS is in `FSDS_code.py`. We provide an explanation and a tutorial for the code. Please refer to `FSDS_explanation.ipynb`.


## Experimental Results
We provide super-resolution results and code for Tab.1.

The super-resolution results can be found in: [https://huggingface.co/RisingEntropy/NNsAreLPFing/blob/main/sr_results.zip](https://huggingface.co/RisingEntropy/NNsAreLPFing/blob/main/sr_results.zip)

Code that generates Tab.1 is in `cal_metrcis_and_table.py`. Please unzip `sr_results.zip` to the `sr_results` directory and run the code. The `sr_results` directory should look like:
```
sr_results
├─ArbSR
│  ├─RCAN_x12
│  ├─RCAN_x2
│  ├─RCAN_x3
│  ├─RCAN_x4
│  ├─RCAN_x6
│  └─RCAN_x8
├─Bicubic
│  ├─Bicubic_x12
│  ├─Bicubic_x18
│  ├─Bicubic_x2
│  ├─Bicubic_x3
│  ├─Bicubic_x4
│  └─Bicubic_x6
├─edsr_baseline
│  ├─x2
│  ├─x3
│  └─x4
├─GRL
│  ├─base
│  │  ├─X2
│  │  ├─X3
│  │  └─X4
│  ├─small
│  │  ├─X2
│  │  ├─X3
│  │  └─X4
│  └─tiny
│      ├─X2
│      ├─X3
│      └─X4
├─HAT
│  ├─HAT-S_SRx2
│  ├─HAT-S_SRx3
│  ├─HAT-S_SRx4
│  ├─HAT_SRx2
│  ├─HAT_SRx3
│  └─HAT_SRx4
├─HDSRNet
│  ├─X2
│  ├─X3
│  └─X4
├─hr
├─ITSRN
│  ├─ITSRN_x12
│  ├─ITSRN_x2
│  ├─ITSRN_x3
│  ├─ITSRN_x4
│  └─ITSRN_x6
├─liif
│  ├─edsr_x12
│  ├─edsr_x18
│  ├─edsr_x2
│  ├─edsr_x3
│  ├─edsr_x4
│  ├─edsr_x6
│  ├─rdn_x12
│  ├─rdn_x18
│  ├─rdn_x2
│  ├─rdn_x3
│  ├─rdn_x4
│  └─rdn_x6
├─LTE
│  ├─EDSR_baseline_x12
│  ├─EDSR_baseline_x2
│  ├─EDSR_baseline_x3
│  ├─EDSR_baseline_x4
│  ├─EDSR_baseline_x6
│  ├─RDN_x12
│  ├─RDN_x2
│  ├─RDN_x3
│  ├─RDN_x4
│  ├─RDN_x6
│  ├─SwinIR_x12
│  ├─SwinIR_x2
│  ├─SwinIR_x3
│  ├─SwinIR_x4
│  └─SwinIR_x6
├─OPESR
│  ├─EDSR_x2
│  ├─EDSR_x3
│  ├─EDSR_x4
│  ├─RDN_x2
│  ├─RDN_x3
│  └─RDN_x4
├─RDN
│  ├─RDN_small_x2
│  ├─RDN_small_x3
│  └─RDN_small_x4
├─SRNO
│  ├─EDSR_baseline_x12
│  ├─EDSR_baseline_x2
│  ├─EDSR_baseline_x3
│  ├─EDSR_baseline_x4
│  └─EDSR_baseline_x6
└─SwinIR
    ├─swinir_classical_sr_x2
    ├─swinir_classical_sr_x3
    ├─swinir_classical_sr_x4
    └─swinir_classical_sr_x8
```