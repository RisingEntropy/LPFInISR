close all;clearvars;
pred = imread('example_figures\EDSRx4.png');
gt = imread('example_figures\gt.png');
FrequencySpectrumDistributionSimilarity(single(pred), single(gt))