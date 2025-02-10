import os
from math import ceil, floor
from os.path import join

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from trimesh.voxel import creation as vox_creation

from augment import random_permute_flat, random_permute_mlp, sorted_permute_mlp
from hd_utils import generate_mlp_from_weights, get_mlp
from siren.dataio import anime_read


class VoxelDataset(Dataset):
    def __init__(
        self, mesh_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        self.mesh_folder = mesh_folder
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))

        self.mesh_files = []
        if object_names is None:
            self.mesh_files = [
                file
                for file in list(os.listdir(mesh_folder))
                if file not in ["train_split.lst", "test_split.lst", "val_split.lst"]
            ]
        else:
            for file in list(os.listdir(mesh_folder)):
                if file.split(".")[0] in blacklist and cfg.filter_bad:
                    continue

                if (
                    ("_" in file and file.split("_")[1] in object_names)
                    or file in object_names
                    or file.split(".")[0] in object_names
                ):
                    self.mesh_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.cfg = cfg
        self.vox_folder = self.mesh_folder + "_vox"
        os.makedirs(self.vox_folder, exist_ok=True)

    def __getitem__(self, index):
        dir = self.mesh_files[index]
        path = join(self.mesh_folder, dir)
        resolution = self.cfg.vox_resolution
        voxel_size = 1.9 / (resolution - 1)
        total_time = self.cfg.unet_config.params.image_size
        if self.cfg.mlp_config.params.move:
            folder_name = os.path.basename(path)
            anime_file_path = os.path.join(path, folder_name + ".anime")
            nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_file_path)

            def normalize(obj, v_min, v_max):
                vertices = obj.vertices
                vertices -= np.mean(vertices, axis=0, keepdims=True)
                vertices *= 0.95 / (max(abs(v_min), abs(v_max)))
                obj.vertices = vertices
                return obj

            # total_time = min(nf, total_time)
            vert_datas = []
            v_min, v_max = float("inf"), float("-inf")

            frames = np.linspace(0, nf, total_time, dtype=int, endpoint=False)
            if self.cfg.move_sampling == "first":
                frames = np.linspace(
                    0, min(nf, total_time), total_time, dtype=int, endpoint=False
                )

            for t in frames:
                vert_data_copy = vert_data
                if t > 0:
                    vert_data_copy = vert_data + offset_data[t - 1]
                vert_datas.append(vert_data_copy)
                vert = vert_data_copy - np.mean(vert_data_copy, axis=0, keepdims=True)
                v_min = min(v_min, np.amin(vert))
                v_max = max(v_max, np.amax(vert))
            grids = []
            for vert_data in vert_datas:
                obj = trimesh.Trimesh(vert_data, face_data)
                obj = normalize(obj, v_min, v_max)
                voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                    obj, pitch=voxel_size
                )
                voxel_grid.fill()
                grid = voxel_grid.matrix
                padding_amounts = [
                    (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                    for length in grid.shape
                ]
                grid = np.pad(grid, padding_amounts).astype(np.float32)
                grids.append(grid)
            grid = np.stack(grids)
        else:
            mesh: trimesh.Trimesh = trimesh.load(path)
            coords = np.asarray(mesh.vertices)
            coords = coords - np.mean(coords, axis=0, keepdims=True)
            v_max = np.amax(coords)
            v_min = np.amin(coords)
            coords *= 0.95 / (max(abs(v_min), abs(v_max)))
            mesh.vertices = coords
            voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                mesh, pitch=voxel_size
            )
            voxel_grid.fill()
            grid = voxel_grid.matrix
            padding_amounts = [
                (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                for length in grid.shape
            ]
            grid = np.pad(grid, padding_amounts).astype(np.float32)

        # Convert 0 regions to -1, so that the input is -1 or +1.
        grid[grid == 0] = -1

        grid = torch.tensor(grid).float()

        # Doing some sanity checks for 4D and 3D generations
        if self.cfg.mlp_config.params.move:
            assert (
                grid.shape[0] == total_time
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
                and grid.shape[3] == resolution
            )
            return grid, 0
        else:
            assert (
                grid.shape[0] == resolution
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
            )

        return grid[None, ...], 0

    def __len__(self):
        return len(self.mesh_files)


class WeightDataset(Dataset):
    def __init__(
        self, mlps_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        ### Path to all the MLP's checkpoints
        ### mlps_folder = ./mlp_weights/3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad
        self.mlps_folder = mlps_folder

        ### condition: 'no' in train_plane.yaml
        self.condition = cfg.transformer_config.params.condition 
        
        ### Get all the files in the MLP's folder
        files_list = list(os.listdir(mlps_folder))

        ### Black list
        blacklist = {}
        if cfg.filter_bad: ### filter_bad: True in train_plane.yaml
            ### Read filter_bad_path: ./data/plane_problematic_shapes.txt
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))

        ### Check the name list
        ### object_names = train_object_names
        ### so will not go to this case
        if object_names is None:
            ### Get all the names from the name of checkpoints
            self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        
        ### Will go to this case
        ### object_names = train_object_names
        else: 
            self.mlp_files = []
            ### Get only the names from the input name list
            for file in list(os.listdir(mlps_folder)):

                ### They exclude black listed shapes:
                ### ≈ 15% of airplane, ≈ 16% of chair and ≈ 51% of car shapes 
                ### in our train split of ShapeNet [1] contain major self-intersections
                # Excluding black listed shapes
                if cfg.filter_bad and file.split("_")[1] in blacklist:
                    continue

                ### Check if file name is correct with their format
                ### For example, file = occ_1a04e3eab45ca15dd86060f189eb133_jitter_0_model_final.pth
                ### and `1a04e3eab45ca15dd86060f189eb133.obj` in object_names
                ### file.split("_")[1] = `1a04e3eab45ca15dd86060f189eb133` in object_names
                # Check if file is in corresponding split (train, test, val)
                # In fact, only train split is important here because we don't use test or val MLP weights
                if ("_" in file and (file.split("_")[1] in object_names or (
                        file.split("_")[1] + "_" + file.split("_")[2]) in object_names)) or (file in object_names):
                    self.mlp_files.append(file)

        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims ### Not used in this code
        self.mlp_kwargs = mlp_kwargs ### Not used in this code

        ### augment: False in train_plane.yaml
        ### so will not go to this case
        if cfg.augment in ["permute", "permute_same", "sort_permute"]: 
            self.example_mlp = get_mlp(mlp_kwargs)

        self.cfg = cfg

        ### --------------- Not used in this code --------------- ###
        ### There is no first_weight_name in train_plane.yaml
        if "first_weight_name" in cfg and cfg.first_weight_name is not None:
            self.first_weights = self.get_weights(
                torch.load(os.path.join(self.mlps_folder, cfg.first_weight_name))
            ).float()

        else: ### Will go to this case
            ### It looks like first_weight_name is not used in this code.
            ### Even it specified in overfit_plane.yaml but not used in this code.
            self.first_weights = torch.tensor([0])

    def get_weights(self, state_dict):
        ### Read all the weights from the state_dict
        weights = []
        shapes = [] ### Not used in this code
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))

            ### Flatten the weight and convert to CPU
            weights.append(state_dict[weight].flatten().cpu())

        ### Then concatenate all the weights into a single tensor
        weights = torch.hstack(weights)

        ### Clone the weights for keeping the original weights
        ### however, it is not used in this code because augment is False
        prev_weights = weights.clone()

        ### --------------- Not used in this code --------------- ###
        # Some augmentation methods are available althougwe don't use them in the main paper
        if self.cfg.augment == "permute":
            weights = random_permute_flat(
                [weights], self.example_mlp, None, random_permute_mlp
            )[0]
        if self.cfg.augment == "sort_permute":
            example_mlp = generate_mlp_from_weights(weights, self.mlp_kwargs)
            weights = random_permute_flat(
                [weights], example_mlp, None, sorted_permute_mlp
            )[0]
        if self.cfg.augment == "permute_same":
            weights = random_permute_flat(
                [weights],
                self.example_mlp,
                int(np.random.random() * self.cfg.augment_amount),
                random_permute_mlp,
            )[0]
        if self.cfg.jitter_augment:
            weights += np.random.uniform(0, 1e-3, size=weights.shape)

        if self.transform:
            weights = self.transform(weights)
        # We also return prev_weights, in case you want to do permutation, we store prev_weights to sanity check later
        return weights, prev_weights

    def __getitem__(self, index):
        ### Get the file name
        file = self.mlp_files[index]

        ### Get the path of the file
        ### mlps_folder = ./mlp_weights/3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad
        ### file = occ_1a04e3eab45ca15dd86060f189eb133_jitter_0_model_final.pth
        dir = join(self.mlps_folder, file)
        

        ### --------------- Check if the file is a directory --------------- ###
        ### Check if the file is a directory
        if os.path.isdir(dir):
            path1 = join(dir, "checkpoints", "model_final.pth")
            path2 = join(dir, "checkpoints", "model_current.pth")
            state_dict = torch.load(path1 if os.path.exists(path1) else path2)
        
        ### will go to this case because file is not a directory
        else:
            state_dict = torch.load(dir, map_location=torch.device("cpu"))

        ### Get the weights from the state_dict
        ### weights_prev is not used in this code because augment is False
        ### weights.shape = (36737,)
        weights, weights_prev = self.get_weights(state_dict)

        ### --------------- Not used in this code --------------- ###
        if self.cfg.augment == "inter":
            other_index = np.random.choice(len(self.mlp_files))
            other_dir = join(self.mlps_folder, self.mlp_files[other_index])
            other_state_dict = torch.load(other_dir)
            other_weights, _ = self.get_weights(other_state_dict)
            lerp_alpha = np.random.uniform(
                low=0, high=self.cfg.augment_amount
            )  # Prev: 0.3
            weights = torch.lerp(weights, other_weights, lerp_alpha)

        ### Return the weights and the previous weights (not used in this code)
        ### And why return 2 previous weights?
        return weights.float(), weights_prev.float(), weights_prev.float()

    def __len__(self):
        return len(self.mlp_files)
