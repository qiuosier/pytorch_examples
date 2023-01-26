import os
import hydra
import torch
from omegaconf import DictConfig
from model import GPT, GPTConfig
from char_dataset import CharDataset, DataConfig


@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    gpt_cfg = GPTConfig(**cfg['gpt_config'])
    data_cfg = DataConfig(**cfg['data_config'])

    # vocab_size and block_size is determined by the training data.
    # They will affect the size of the GPT model
    # data_cfg.truncate = 1
    # data_cfg.block_size = 1024
    dataset = CharDataset(data_cfg)

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    # gpt_cfg.vocab_size = 50257
    gpt_cfg.n_layer = 12
    gpt_cfg.n_head = 12
    gpt_cfg.n_embd = 768

    settings = [
        (8, 8, 512),
        (8, 8, 768),
        (12, 8, 768),
        (12, 12, 768),
        (12, 8, 1024),
        (16, 8, 1024),
        (16, 16, 1024),
        (24, 8, 1024),
        (24, 16, 1024),
        (24, 32, 1024),
        (36, 16, 1024),
        (36, 20, 1280),
    ]

    model_file_name = "model.pt"

    for gpt_cfg.n_layer, gpt_cfg.n_head, gpt_cfg.n_embd in settings:
        print("=" * 50)
        print(gpt_cfg)
        model = GPT(gpt_cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters in model: %.2fM" % (n_params/1e6,))
        torch.save(model.state_dict(), model_file_name)
        size_in_mb = os.path.getsize(model_file_name) / 1024 / 1024
        print("Model state dict size: %.2fMB" % size_in_mb)
        os.remove(model_file_name)


if __name__ == "__main__":
    main()
