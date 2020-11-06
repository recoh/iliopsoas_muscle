# Large-Scale Analysis of Iliopsoas Muscle Volumes in the UK Biobank

If you use the code or the model in your work, please cite:
- Fitzpatrick J, Basty N, Cule M, Liu Y, Bell JD, Thomas EL, Whitcher B. Large-Scale Analysis of Iliopsoas Muscle Volumes in the UK Biobank, 2020, [arXiv:2008.05217][arxiv-ipm].

## Installation

Execute `git-lfs checkout` in the directory containing the `git clone` output to download the deep learning models and example data.

```
# deactivate
# rm -rf ~/.venv/ipm
python3 -m venv ~/.venv/ipm 
source ~/.venv/ipm/bin/activate 
pip install --upgrade pip setuptools wheel 
cd ~/iliopsoas_muscle
pip install -r requirements.txt 
jupyter notebook 
```

Click on the link and then click on the `.ipynb` file to run the example in the browser.

## Pre-processing

The example subject from the UK Biobank was pre-processed using our [pipeline][pipeline] code.  
- Basty, N., Yiu, L., Cule, M., Thomas, E.L., Bell, J.D., Whitcher, B.  Image processing and quality control for 
abdominal magnetic resonance imaging in the UK Biobank, 2020, [arXiv:2007.01251][arxiv-pipeline].

[arxiv-ipm]: https://arxiv.org/abs/2008.05217
[arxiv-pipeline]: https://arxiv.org/abs/2007.01251
[pipeline]: https://github.com/recoh/pipeline
