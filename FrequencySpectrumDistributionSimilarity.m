function scores = FrequencySpectrumDistributionSimilarity(pred, gt)
    pyenv('Version','C:\Users\RisingEntropy\scoop\apps\miniconda3-py311\current\envs\CV\python.exe');
    module = py.importlib.import_module('FSDS_code');
    scores = module.matlab_interface(single(pred), single(gt));
end
