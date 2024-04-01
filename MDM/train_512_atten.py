from aten_64 import Unet, GaussianDiffusion, Trainer

num_classes = 1
num_steps = 1000
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), num_classes=1, cond_drop_prob=0)

diffusion = GaussianDiffusion(
    model,
    image_size=(384, 512),
    timesteps=num_steps,
    sampling_timesteps=2,
    beta_schedule="linear",
    objective="pred_x0",
)

trainer = Trainer(
    diffusion,
    "../DIR-D/training",
    "../DIR-D/testing",
    real_image_size=(384, 512),
    augment_horizontal_flip=False,
    train_batch_size=64,
    val_batch_size=32,
    train_lr=2e-4,
    train_num_steps=500000,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    amp=False,
    calculate_fid=False,
    num_samples=16,
    save_and_sample_every=5000,
    results_folder="./result_512",
)

trainer.train()
