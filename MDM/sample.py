import pathlib
from aten_64 import Unet, GaussianDiffusion, Trainer

if __name__ == "__main__":
    image_size = (384, 512)
    sampling_timesteps = 2
    result_folder = "../DIR-D/testing"

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), num_classes=1, cond_drop_prob=0)

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        beta_schedule="linear",
        objective="pred_x0",
    )

    trainer = Trainer(
        diffusion,
        "../DIR-D/training",
        "../DIR-D/testing",
        real_image_size=(384, 512),
        train_batch_size=64,
        val_batch_size=4,
        train_lr=3e-4,
        train_num_steps=150000,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False,
        num_samples=16,
        save_and_sample_every=5000,
        results_folder=result_folder,
    )

    trainer.load("../Checkpoints/MDM.pth")

    trainer.sample(pathlib.Path(result_folder), "cuda")
