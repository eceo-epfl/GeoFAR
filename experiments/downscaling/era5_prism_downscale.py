# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from climate_learn.models.hub import VisionTransformer, Interpolation, Unet, ResNet, EDSR, SwinIR, SRFormer, Constraint_ViT, DeepSD, DSFNO, Generator, Discriminator, EDMPrecond, \
    GeoFAR, GeoFAR_Unet, GeoFAR_Generator, GeoFAR_DSFNO
import datetime
import torch.nn as nn
import torch

from climate_learn.transforms import Mask, Denormalize
import numpy as np
import torch.nn.functional as F
import os

parser = ArgumentParser()
parser.add_argument("low_res_dir")
parser.add_argument("high_res_dir")
parser.add_argument("preset")
parser.add_argument("--bs", type=int, default=32) # batch size
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50) # max training epochs
parser.add_argument("--patience", type=int, default=10) 
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None) # specify it for testing
parser.add_argument("--lr", default=1e-5, type=float) # learning rate
parser.add_argument("--t_res", default=24, type=int, help="The temporal resolution (in hours) of the dataset, used for diffusion based methods.")
args = parser.parse_args()
input_size = (32, 64) 
target_size = (32, 64) 
patch_size = 2

# Set up data
# Set up data
dm = cl.data.ERA5toPRISMDataModule(
    args.era5_cropped_dir,
    args.prism_processed_dir,
    batch_size=args.bs,
    num_workers=4,
)
dm.setup()

img_num = 12800

# Set up masking
mask = Mask(dm.get_out_mask().to(device=f"cuda:{args.gpu}"))
denorm = Denormalize(dm)
denorm_mask = lambda x: denorm(mask(x))

### format the 'mask'
dem_path = '/home/xc/climate_datasets/prism_oro'
dem = torch.from_numpy(np.load(os.path.join(dem_path, "orography.npz"))['data'])
padded_dem = F.pad(dem, (2, 2, 3, 3))
dem_mask = torch.from_numpy(np.load(os.path.join(dem_path, "mask.npy")))
padded_mask = F.pad(dem_mask, (2, 2, 3, 3))
dem = torch.where(padded_mask == 1, padded_dem, 0)
###

# learning rate, patch size, training configurations, loss

# Set up deep learning model
if args.preset == "vit":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        VisionTransformer(
            img_size=target_size, 
            in_channels=1,
            out_channels=1,
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask],
    )
elif args.preset == "unet":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        Unet(
            in_channels=1,  
            out_channels=1,  
            hidden_channels=64,  
            n_blocks=2,  
            ch_mults=[1, 2, 2],  
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "resnet":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        ResNet(
            in_channels=1,  
            out_channels=1,  
            hidden_channels=128, 
            n_blocks=28,  
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "edsr":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        EDSR(
            in_channels=1,  
            out_channels=1,  
            n_resblocks=28, 
            n_feats=128, 
            scale=1, 
            n_colors=1, 
            res_scale=0.1,
        ),
    )
    optim_kwargs = {"lr": 1e-8, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "ffl":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        VisionTransformer(
            img_size=target_size, 
            in_channels=1,
            out_channels=1,
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "ffl",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "swinir":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        SwinIR(
            in_chans=1,
            upscale=1, 
            img_size=input_size, # You need to replace to the real input size
            window_size=2, 
            img_range=1., 
            depths=[3, 3, 3, 3],
            embed_dim=128, 
            num_heads=[4, 4, 4, 4], 
            mlp_ratio=2, 
            upsampler='pixelshuffledirect'),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "srformer":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        SRFormer(
            in_chans=1,
            upscale=1, 
            img_size=input_size,  # You need to replace to the real input size
            window_size=patch_size, 
            img_range=1., 
            depths=[3, 3, 3, 3],
            embed_dim=128, 
            num_heads=[4, 4, 4, 4], 
            mlp_ratio=2, 
            upsampler='pixelshuffledirect',
            resi_connection= '1conv'
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "deepsd":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        DeepSD(
            in_channels=2,  
            out_channels=1,  
            oro_path=dem,
            upscale_factor=1,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "facl":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        VisionTransformer(
            img_size=target_size, 
            in_channels=1,
            out_channels=1,
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "facl",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        total_step=20000,
        micro_batch=args.bs,
    )
elif args.preset == "smcl_vit":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        Constraint_ViT(
            img_size=target_size, 
            in_channels=1,
            out_channels=1,
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
            constraints='softmax',
            upsampling_factor=1,            
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == 'dsfno':
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        DSFNO(
            in_channel=1,
            n_channels=64,
            n_residual_blocks=4,
            n_operator_blocks=2,
            modes=18,
            apply_constraint=True,
            upsample_factor=1,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        elevation=dem,
    )
elif args.preset == "srgan":
    model_g = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        Generator(n_residual_blocks=16, 
                  upsample_factor=1, 
                  base_filter=64, 
                  num_channel=1)
    )
    model_d = nn.Sequential(
        Discriminator(base_filter=64, 
                      num_channel=1)
    )
    srgan_lr = 1e-5
    optim_g = torch.optim.Adam(model_g.parameters(), lr=srgan_lr, betas=(0.9, 0.999))
    optim_d = torch.optim.SGD(model_d.parameters(), lr=srgan_lr, momentum=0.9, nesterov=True)
    model = cl.load_gen_module(
        task='downscaling',
        data_module=dm,
        model_g=model_g,
        model_d=model_d,
        optim_g=optim_g,
        optim_d=optim_d, 
        sched_g=torch.optim.lr_scheduler.MultiStepLR(optim_g, milestones=[25, 35, 45], gamma=0.5), # to do
        sched_d=torch.optim.lr_scheduler.MultiStepLR(optim_d, milestones=[25, 35, 45], gamma=0.5), # to do
        train_lossG=nn.MSELoss(),
        train_lossD=nn.BCELoss(),
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        warmup_epochs=10,
    )
elif args.preset == "climatediffuse":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        EDMPrecond(
            img_resolution=target_size, #
            in_channels=2,
            out_channels=1,
            label_dim = 2,                # Number of class labels, 0 = unconditional.
            use_fp16 = False,            # Execute the underlying model at FP16 precision?
            sigma_min = 0,                # Minimum supported noise level.
            sigma_max = float('inf'),     # Maximum supported noise level.
            sigma_data = 1.0,              # Expected standard deviation of the training data
            model_type = 'UNet',   # Class name of the underlying model.
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_diffusion_module(
        task='downscaling',
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        scaler = torch.cuda.amp.GradScaler(),
        train_loss= "edmloss",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        t_hours = args.t_res,
    )
elif args.preset == 'geofar_vit':
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        GeoFAR(
            img_size=target_size, #
            in_channels=1,
            out_channels=1,
            history=1,
            n_coeff=64,
            n_sh_coeff=64,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2,
            num_heads=4,
            oro_path=dem,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss='mse',
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"],
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        elevation=dem,
    )
elif args.preset == "geofar_unet":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        GeoFAR_Unet(
            img_size=target_size,
            in_channels=1,  # Same as ViT
            out_channels=1,  # Same as ViT
            history=1,
            n_coeff=64,
            n_sh_coeff=64,
            hidden_channels=64,  # Define the base number of channels
            n_blocks=2,  # U-Net depth (adjust as needed)
            ch_mults=[1, 2, 2],  # Channel multipliers for each stage
            oro_path=dem,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)} #0.95
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss='mse',
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
    )
elif args.preset == "geocd_srgan":
    model_g = nn.Sequential(
        Interpolation((target_size, "bilinear")),
        GeoFAR_Generator(n_residual_blocks=16, 
                  upsample_factor=1, 
                  base_filter=64, 
                  num_channel=1,
                  img_size=target_size,
                  oro_path=dem,
                  n_coeff=64, 
                  n_sh_coeff=64)
    )
    model_d = nn.Sequential(
        Discriminator(base_filter=64, 
                      num_channel=1)
    )
    srgan_lr = 1e-5
    optim_g = torch.optim.Adam(model_g.parameters(), lr=srgan_lr, betas=(0.9, 0.999))
    optim_d = torch.optim.SGD(model_d.parameters(), lr=srgan_lr, momentum=0.9, nesterov=True)
    model = cl.load_gen_module(
        task='downscaling',
        data_module=dm,
        model_g=model_g,
        model_d=model_d,
        optim_g=optim_g,
        optim_d=optim_d, 
        sched_g=torch.optim.lr_scheduler.MultiStepLR(optim_g, milestones=[10, 15, 18], gamma=0.5), # to do
        sched_d=torch.optim.lr_scheduler.MultiStepLR(optim_d, milestones=[10, 15, 18], gamma=0.5), # to do
        train_lossG=nn.MSELoss(),
        train_lossD=nn.BCELoss(),
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        warmup_epochs=10,
    )
elif args.preset == "geofar_diffuse":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        EDMPrecond(
            img_resolution=target_size, 
            in_channels=2, # concatenate input and noisy image
            out_channels=1,
            label_dim = 2,                # Number of class labels, 0 = unconditional.
            use_fp16 = False,            # Execute the underlying model at FP16 precision?
            sigma_min = 0,                # Minimum supported noise level.
            sigma_max = float('inf'),     # Maximum supported noise level.
            sigma_data = 1.0,              # Expected standard deviation of the training data
            model_type = 'GeoFAR_Diffuse',   # Class name of the underlying model.
            n_coeff = 64,
            n_sh_coeff = 64,
            oro_path = dem,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_diffusion_module(
        task='downscaling',
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        scaler = torch.cuda.amp.GradScaler(),
        train_loss= "edmloss",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        t_hours = args.t_res,
    )
elif args.preset == 'geofar_dsfno':
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        GeoFAR_DSFNO(
            img_size=target_size,
            in_channel=1,
            n_channels=64,
            n_residual_blocks=4,
            n_operator_blocks=2,
            modes=18,
            apply_constraint=True,
            upsample_factor=1,
            n_coeff=64,
            n_sh_coeff=64,
            oro_path=dem,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
         "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        train_target_transform=mask,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask], 
        elevation=dem,
    )
else:
    model = cl.load_downscaling_module(data_module=dm, architecture=args.preset) # architecture refers to predefined models in CL

# Setup trainer
current_time = datetime.datetime.now().strftime("%m%d-%H-%M")
pl.seed_everything(0)
default_root_dir = f"{args.preset}_downscaling_era5_prism_{current_time}"
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "val/mse:aggregate"

if "diffuse" in args.preset:
    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=args.summary_depth),
        ModelCheckpoint(
            dirpath=f"{default_root_dir}/checkpoints",
            monitor=early_stopping,
            filename="epoch_{epoch:03d}",
            auto_insert_metric_name=False,
            save_last=True
        ),
    ]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        accelerator="gpu" if args.gpu != -1 else None,
        devices=[args.gpu] if args.gpu != -1 else None,
        max_epochs=args.max_epochs,
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
        limit_val_batches=0,
        accumulate_grad_batches= 4,
    )
else:
    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=args.summary_depth),
        EarlyStopping(monitor=early_stopping, patience=args.patience),
        ModelCheckpoint(
            dirpath=f"{default_root_dir}/checkpoints",
            monitor=early_stopping,
            filename="epoch_{epoch:03d}",
            auto_insert_metric_name=False,
        ),
    ]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        accelerator="gpu" if args.gpu != -1 else None,
        devices=[args.gpu] if args.gpu != -1 else None,
        max_epochs=args.max_epochs,
        strategy="ddp_find_unused_parameters_true",
        precision="16",
    )

# Train and evaluate model from scratch
if args.checkpoint is None:
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
# Evaluate saved model checkpoint
else:
    if 'srgan' in args.preset:
        model = cl.LitSRGAN.load_from_checkpoint(
            args.checkpoint,
            netG=model.netG,
            netD=model.netD,
            optimizerG=model.optimizerG,
            optimizerD=model.optimizerD,
            schedulerG=None,
            schedulerD=None,
            train_lossG=None,
            train_lossD=None,
            val_loss=None,
            test_loss=model.test_loss,
            test_target_transforms=model.test_target_transforms,
        )
    elif 'diffuse' in args.preset:
        model = cl.LitDiffusion.load_from_checkpoint(
            args.checkpoint,
            net=model.net,
            optimizer=model.optimizer,
            lr_scheduler=None,
            train_loss=None,
            val_loss=None,
            test_loss=model.test_loss,
            test_target_transforms=model.test_target_transforms,
        )
    else:
        model = cl.LitModule.load_from_checkpoint(
            args.checkpoint,
            net=model.net,
            optimizer=model.optimizer,
            lr_scheduler=None,
            train_loss=None,
            val_loss=None,
            test_loss=model.test_loss,
            test_target_transforms=model.test_target_transforms,
        )
    trainer.test(model, datamodule=dm)
