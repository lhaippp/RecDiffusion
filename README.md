# [CVPR2024] RecDiffusion: Rectangling for Image Stitching with Diffusion Models
[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_RecDiffusion_Rectangling_for_Image_Stitching_with_Diffusion_Models_CVPR_2024_paper.pdf)

## How to Sample And Calculate Metrics

1. Create conda environment (`environment.yaml`)
2. Download DIR-D dataset, pseudo mesh and checkpoints from [HuggingFace](https://huggingface.co/pure01fx/RecDiffusion)
3. Extract dataset. Now your root directory should contains: `/CDM`, `/MDM`, `/DIR-D`, `/Checkpoints`
4. Run "cd MDM && python sample.py" to generate MDM intermediate result
5. Run "cd CDM && python sample.py" to generate final result
6. Run "python metric.py" to calculate metrics

## How to Train

1. A lower version of `pytorch-lightning` is needed. Install environment from `environment-training.yaml`. (Tested using `micromamba`, if installing by conda is failed, consider manually install all packages in this file.)
2. Train MDM first: `cd MDM && accelerate launch train_512_atten.py`. You may want to modify this file to change batch size, etc. Please refer to `accelerate`'s documents for more information.
3. When training is completed, modify `MDM/sample.py`. Specifically, replace `testing` with `training` and change the path to your checkpoint.
4. Train CDM: `cd CDM && python main.py fit -b configs/rectangling.yaml`

## Citations
```
@inproceedings{zhou2024recdiffusion,
  title={RecDiffusion: Rectangling for Image Stitching with Diffusion Models},
  author={Zhou, Tianhao and Li, Haipeng and Wang, Ziyi and Luo, Ao and Zhang, Chen-Lin and Li, Jiajun and Zeng, Bing and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2692--2701},
  year={2024}
}
```
