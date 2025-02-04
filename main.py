import os

from dataset import VoxelDataset, WeightDataset
from hd_utils import Config, get_mlp
from hyperdiffusion import HyperDiffusion

# Using it to make pyrender work on clusters
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
from datetime import datetime
from os.path import join

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import ldm.ldm.modules.diffusionmodules.openaimodel
import wandb
from transformer import Transformer

sys.path.append("siren")


@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="train_plane",
)
def main(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method") ### Get method: hyper_3d
    mlp_kwargs = None

    ### Get the MLP configuration
    # In HyperDiffusion, we need to know the specifications of MLPs that are used for overfitting
    if "hyper" in method: ### method: hyper_3d
        mlp_kwargs = Config.config["mlp_config"]["params"]

    ### Initialize WandB
    wandb.init(
        project="hyperdiffusion",
        dir=config["tensorboard_log_dir"],
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        tags=[Config.get("mode")],
        mode="disabled" if Config.get("disable_wandb") else "online",
        config=dict(config),
    )

    wandb_logger = WandbLogger()
    wandb_logger.log_text("config", ["config"], [[str(config)]])
    print("wandb", wandb.run.name, wandb.run.id)

    ### Initialize datasets
    train_dt = val_dt = test_dt = None

    ### Path to all the MLP's checkpoints
    ### E.g., mlps_folder_train = ./mlp_weights/3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad
    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    mlps_folder_train = Config.get("mlps_folder_train")

    ### --------- Get the diffusion model --------- ###
    # Initialize Transformer for HyperDiffusion
    ### method: hyper_3d
    if "hyper" in method:
        ### Get the MLP from the MLP's configuration
        mlp = get_mlp(mlp_kwargs)

        ### Flatten the MLP's weights
        state_dict = mlp.state_dict()
        layers = []
        layer_names = []
        for l in state_dict:
            shape = state_dict[l].shape
            layers.append(np.prod(shape)) ### Get the number of weights in each layer
            layer_names.append(l)

        ### Create diffusion model
        model = Transformer(
            layers, layer_names, **Config.config["transformer_config"]["params"]
        ).cuda()
    # Initialize UNet for Voxel baseline
    else:
        model = ldm.ldm.modules.diffusionmodules.openaimodel.UNetModel(
            **Config.config["unet_config"]["params"]
        ).float()

    ### Get the path to the dataset list: train_split.lst, val_split.lst, test_split.lst
    ### E.g., dataset_path = ./data/02691156
    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])

    ### Get the object names in the train_split.lst
    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )

    ### If static MLPs are used, remove the .obj extension
    ### E.g., e4665d76bf8fc441536d5be52cb9d26a.obj -> e4665d76bf8fc441536d5be52cb9d26a
    if not cfg.mlp_config.params.move:
        train_object_names = set([str.split(".")[0] for str in train_object_names])

    ### -------------------------------------------------------------------------- Dataset -------------------------------------------------------------------------- ###
    # Check if dataset folder already has train,test,val split; create otherwise.
    ### method = "hyper_3d" in train_plane.yaml
    if method == "hyper_3d": 
        mlps_folder_all = mlps_folder_train

        ### --------- Split the dataset --------- ###
        ### This is how they split the dataset into train(80%), val(5%), test(15%)

        ### Get all the object names besides `.lst` files
        ### Properly empty because current dataset_path directory only contains .lst files
        all_object_names = np.array(
            [obj for obj in os.listdir(dataset_path) if ".lst" not in obj]
        )
        total_size = len(all_object_names)  ### 0
        val_size = int(total_size * 0.05)   ### 0
        test_size = int(total_size * 0.15)  ### 0
        train_size = total_size - val_size - test_size  ### 0

        ### If the train_split.lst does not exist, create it
        if not os.path.exists(os.path.join(dataset_path, "train_split.lst")):
            ### Random choice in `range(total_size)` with size `train_size + val_size`
            ### No duplication
            train_idx = np.random.choice(
                total_size, train_size + val_size, replace=False
            )

            ### Get the remain that not in train_valid
            test_idx = set(range(total_size)).difference(train_idx)

            ### Get the valid in the train_valid
            val_idx = set(np.random.choice(train_idx, val_size, replace=False))

            ### The remain in train_valid
            train_idx = set(train_idx).difference(val_idx)
            print(
                "Generating new partition",
                len(train_idx),
                train_size,
                len(val_idx),
                val_size,
                len(test_idx),
                test_size,
            )

            # Sanity checking the train, val and test splits
            assert len(train_idx.intersection(val_idx.intersection(test_idx))) == 0
            assert len(train_idx.union(val_idx.union(test_idx))) == total_size
            assert (
                len(train_idx) == train_size
                and len(val_idx) == val_size
                and len(test_idx) == test_size
            )

            np.savetxt(
                os.path.join(dataset_path, "train_split.lst"),
                all_object_names[list(train_idx)],
                delimiter=" ",
                fmt="%s",
            )
            np.savetxt(
                os.path.join(dataset_path, "val_split.lst"),
                all_object_names[list(val_idx)],
                delimiter=" ",
                fmt="%s",
            )
            np.savetxt(
                os.path.join(dataset_path, "test_split.lst"),
                all_object_names[list(test_idx)],
                delimiter=" ",
                fmt="%s",
            )

        ### --------- Get the object names in the *.lst files --------- ###
        val_object_names = np.genfromtxt(
            os.path.join(dataset_path, "val_split.lst"), dtype="str"
        )
        val_object_names = set([str.split(".")[0] for str in val_object_names])
        test_object_names = np.genfromtxt(
            os.path.join(dataset_path, "test_split.lst"), dtype="str"
        )
        test_object_names = set([str.split(".")[0] for str in test_object_names])
        # assert len(train_object_names) == train_size, f"{len(train_object_names)} {train_size}"

        ### train_object_names is already readed above
        ### So weird that they read it first, then create the lst files ...
        train_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            train_object_names,
        )
        train_dl = DataLoader(
            train_dt,
            batch_size=Config.get("batch_size"), # batch_size: 32
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        val_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            val_object_names,
        )
        test_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            test_object_names,
        )
    elif method == "raw_3d":
        dataset_path = os.path.join(
            Config.config["dataset_dir"], Config.config["dataset"]
        )
        train_dt = VoxelDataset(
            dataset_path, wandb_logger, model.dims, mlp_kwargs, cfg, train_object_names
        )
        train_dl = DataLoader(
            train_dt, batch_size=Config.get("batch_size"), shuffle=True, num_workers=2
        )

    # These two dl's are just placeholders, during val and test evaluation we are looking at test_split.lst,
    # val_split.lst files, inside calc_metrics methods
    val_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]), batch_size=1, shuffle=False
    )
    test_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]), batch_size=1, shuffle=False
    )

    print(
        "Train dataset length: {} Val dataset length: {} Test dataset length".format(
            len(train_dt), len(val_dt), len(test_dt)
        )
    )
    input_data = next(iter(train_dl))[0]
    print(
        "Input data shape, min, max:",
        input_data.shape,
        input_data.min(),
        input_data.max(),
    )
    ### -------------------------------------------------------------------------- End of Dataset -------------------------------------------------------------------------- ###


    best_model_save_path = Config.get("best_model_save_path")
    model_resume_path = Config.get("model_resume_path")

    
    # Initialize HyperDiffusion
    ### input_data.shape = [B, n_weight]
    diffuser = HyperDiffusion(
        model, train_dt, val_dt, test_dt, mlp_kwargs, input_data.shape, method, cfg
    )

    # Specify where to save checkpoints
    checkpoint_path = join(
        config["tensorboard_log_dir"],
        "lightning_checkpoints",
        f"{str(datetime.now()).replace(':', '-') + '-' + wandb.run.name + '-' + wandb.run.id}",
    )
    best_acc_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/1-NN-CD-acc",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-nn-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    )

    best_mmd_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/lgan_mmd-CD",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-mmd-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    )

    last_model_saver = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="last-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
        save_on_train_epoch_end=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=Config.get("epochs"),
        strategy="ddp",
        logger=wandb_logger,
        default_root_dir=checkpoint_path,
        callbacks=[
            best_acc_checkpoint,
            best_mmd_checkpoint,
            last_model_saver,
            lr_monitor,
        ],
        check_val_every_n_epoch=Config.get("val_fid_calculation_period"),
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    if Config.get("mode") == "train":
        # If model_resume_path is provided (i.e., not None), the training will continue from that checkpoint
        trainer.fit(diffuser, train_dl, val_dl, ckpt_path=model_resume_path)

    # best_model_save_path is the path to saved best model
    trainer.test(
        diffuser,
        test_dl,
        ckpt_path=best_model_save_path if Config.get("mode") == "test" else None,
    )
    wandb_logger.finalize("Success")


if __name__ == "__main__":
    main()
