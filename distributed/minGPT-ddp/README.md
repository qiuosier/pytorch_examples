# minGPT-DDP

Code accompanying the tutorial at https://pytorch.org/tutorials/intermediate/ddp_minGPT.html for training a GPT-like model with Distributed Data Parallel (DDP) in PyTorch. 

Files marked with an asterisk (*) are adapted from the minGPT repo (https://github.com/karpathy/minGPT). 

- [trainer.py](mingpt/trainer.py) includes the Trainer class that runs the distributed training iterations on the model with the provided dataset.
- [model.py *](mingpt/model.py) defines the model architecture.
- [char_dataset.py *](mingpt/char_dataset.py) contains the `Dataset`class for a character-level dataset.
- [gpt2_train_cfg.yaml](mingpt/gpt2_train_cfg.yaml) contains the configurations for data, model, optimizer and training run.
- [main.py](mingpt/main.py) is the entry point to the trainig job. It sets up the DDP process group, reads all the configurations and runs the training job.
- [slurm/](mingpt/slurm) contains files for setting up an AWS cluster and the slurm script to run multinode training.
- [model_size.py](mingpt/model_size.py) is a script to calculate the number of parameters and the file size of the models.

Command line arguments can be used to modified the configurations. See [gpt2_train_cfg.yaml](mingpt/gpt2_train_cfg.yaml) for defaults and available options. For example, to change the batch size, run

```
python main.py --trainer_config.batch_size=128
```

To use [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) for training the model:

```
python main.py --trainer_config.distributed_method=fsdp
```

To enable cpu offload when training with FSDP:
```
python main.py --trainer_config.distributed_method=fsdp --trainer_config.cpu_offload=True
```

Compare with main to see the code changes: https://github.com/qiuosier/pytorch_examples/compare/main...qiuosier:pytorch_examples:minGPT
