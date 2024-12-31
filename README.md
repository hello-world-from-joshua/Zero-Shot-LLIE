<div align="center">

<h2>Zero-Shot Low Light Image Enhancement with Diffusion Prior</h2>

<div>
    <a href='https://github.com/hello-world-from-joshua' target='_blank'>Joshua Cho</a>&emsp;
    <a href='https://saraaghajanzadeh.github.io/' target='_blank'>Sara Aghajanzadeh</a>&emsp;
    <a href='https://zzhu.vision/' target='_blank'>Zhen Zhu</a>&emsp;
    <a href='http://luthuli.cs.uiuc.edu/~daf/' target='_blank'>D. A. Forsyth</a>
</div>
<div>
    University of Illinois Urbana-Champaign&emsp; 
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2412.13401" target='_blank'>[arXiv]</a>
    </h4>
</div>


</div>

## :mega: Updates
:star: **12.19.2024**: We plan to have the code available in late December 2024. (tentative)

:star: **12.31.2024**: Code is released.


## :sunny: Results
### :pushpin: Please Read
- All results are provided in the **`results/link.txt`** to ensure transparency of our paper, despite the superior performance of our method being evident through quantitative metrics.
- Please review the results for LOLv2 first **`(results/ 0_lolv2test.pdf & 1_lolv2train.pdf)`**, as it includes more images of the same scene under varying brightness and noise levels, before proceeding to LOLv1 and unpaired datasets (DICM, MEF, LIME, NPE, and VV). Given the zero-shot framework of our method, we also provide results on the training set.
- A robust method should fulfill the following criteria:

1. **Consistency Across Conditions**: The method output should remain identical for the same scene, regardless of variations in darkness and noise levels (e.g., Figure 10 in our paper).

2. **Beyond Naive Scaling**: 
   - (a) The method must identify the intrinsic colors of the scene, overcoming incorrect colors introduced by noise (e.g., row 2 in Figure 7 of the motorcycle scene).
   - (b) It should perform region-specific brightness adjustments instead of uniform global scaling, which can result in unnatural appearances (e.g., row 2 in Figure 9). Observe how other methods resemble uniform global scaling.
   - (c) It should attenuate noise. Please zoom in to examine the noise in the scaled image and the residual noise for each method.

3. **Generality Across Datasets**: The method must perform effectively across diverse datasets and not be limited to excelling in only one.

4. **Unpaired Datasets**: For unpaired datasets, DICM, MEF, LIME, NPE, and VV, which are not dark and contain negligible noise, uniform global scaling is often sufficient to achieve optimal results.

5. **Miscellaneous**: The "scaled image" refers to an image uniformly globally scaled such that its average brightness is 120.0. On average, GDP requires approximately **`19 min/img`** on NVIDIA A10, while our method takes only **`1.4 min/img`** with no additional processes running. In addition, GDP output varies on each run as it is not fixed.


## :desktop_computer: Setup
```
# git clone this repository
git clone https://github.com/hello-world-from-joshua/Zero-Shot-LLIE.git
cd Zero-Shot-LLIE

# create a conda env
conda create -n Zero-Shot-LLIE python=3.9.20 -y
conda activate Zero-Shot-LLIE

# install python dependencies
pip install -r requirements.txt
```

### Data
- Paired Datasets: [here](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
- Unpaired Datasets: [here](https://daooshee.github.io/BMVC2018website/)
  
### Note
To address potential compatibility issues, we provide exact library versions in conda_env/authors_conda_env.txt. If any problems arise, please use this file as a guide for troubleshooting.


## 🚀 Launch
```
python main.py --img_dir_path "./small_sample_data or your_data_path"
```
### Note
Our method relies heavily on self-reconstruction by the diffusion model. Should the output images fall short of expectations, please evaluate the self-reconstruction of the input image in the ./output/LoRA_Reconstructed. If the self-reconstruction is suboptimal, consider increasing the --lora_batch_size and --lora_steps parameters in the command-line configuration. All results in both our paper and the PDF files are obtained using the default configuration, without increasing the two LoRA parameters, to maintain a balance between time and accuracy.


## Acknowledgment
Our work is inspired by [**ControlNet**](https://arxiv.org/pdf/2302.05543) and [**Plug-and-Play**](https://pnp-diffusion.github.io/), and our implementation builds upon the accompanying code.


## Citation
```
@article{cho2024llie,
    title={Zero-Shot Low Light Image Enhancement with Diffusion Prior},
    author={Joshua Cho and Sara Aghajanzadeh and Zhen Zhu and D. A. Forsyth},
    journal={arxiv},
    year={2024}
}
```
