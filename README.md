# [CVPR2024] RecDiffusion: Rectangling for Image Stitching with Diffusion Models

https://arxiv.org/abs/2403.19164

## How to Sample And Calculate Metrics

1. Create conda environment (`environment.yaml`)
2. Download DIR-D dataset, pseudo mesh and checkpoints from [HuggingFace](https://huggingface.co/pure01fx/RecDiffusion)
3. Extract dataset. Now your root directory should contains: `/CDM`, `/MDM`, `/DIR-D`, `/Checkpoints`
4. Run "cd MDM && python sample.py" to generate MDM intermediate result
5. Run "cd CDM && python sample.py" to generate final result
6. Run "python metric.py" to calculate metrics
