
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from ldm.util import instantiate_from_config

cfgs = [ './configs/rectangling.yaml', './configs/test.yaml' ]

if __name__ == '__main__':
    configs = [OmegaConf.load(cfg) for cfg in cfgs]
    config = OmegaConf.merge(*configs)

    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    trainer = Trainer()
    print(f"Loading checkpoint from {config.model.params.ckpt_path}")
    model = instantiate_from_config(config.model)
    trainer.test(model, data)
