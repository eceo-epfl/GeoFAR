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

parser = ArgumentParser()
parser.add_argument("low_res_dir")
parser.add_argument("high_res_dir")
parser.add_argument("preset")
parser.add_argument(
    "variable", choices=["t2m", "z500", "t850", "sp", "10u", "all_surf"], help="The variable to predict."
)  
parser.add_argument("--ratio", type=int, default=2, help="The downscaling ratio (e.g., 2 for 267->534, 4 for 267->1068, 8 for 133->1068).")
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--t_res", default=3, type=int, help="The temporal resolution (in hours) of the dataset, used for diffusion based methods.")
args = parser.parse_args()
input_size = (267, 267) if args.ratio == 2 or args.ratio == 4 else (133, 133) # 2x: 267->534, 4x: 267->1068, 8x: 133->1068
target_size = (534, 534) if args.ratio == 2 else (1068, 1068) # 2x: 267->534, 4x: 267->1068, 8x: 133->1068
patch_size = 6 if args.ratio == 2 else 12

ORO_PATH = '/home/chang/climate_datasets/cerra_subset_oro/orography-534.npz' # the elevation information, change according to target size

# Set up data
var_dict = {
    "t2m": "2m_temperature",
    "rh2m": "2m_relative_humidity",
    "sp": "surface_pressure",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "z500": "geopotential_500",
    "t850": "temperature_850",
}

surf_vars = [
    "2m_temperature",
    "2m_relative_humidity",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

if args.variable == "all_surf":
    in_vars = surf_vars
    out_vars = surf_vars
else:
    full_name = var_dict[args.variable]
    in_vars = [full_name]
    out_vars = [full_name]

dm = cl.data.IterDataModule(
    "downscaling",
    args.low_res_dir,
    args.high_res_dir,
    in_vars,
    out_vars,
    subsample=1,
    batch_size=args.bs,
    buffer_size=2000,
    num_workers=4,
)
dm.setup()
img_num = cl.utils.get_img_num(args.low_res_dir)

# learning rate, patch size, training configurations, loss

# Set up deep learning model
if args.preset == "vit":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        VisionTransformer(
            img_size=target_size, 
            in_channels=len(in_vars),
            out_channels=len(out_vars),
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "unet":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        Unet(
            in_channels=len(in_vars),  
            out_channels=len(out_vars),  
            hidden_channels=64,  
            n_blocks=2,  
            ch_mults=[1, 2, 2],  
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "resnet":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        ResNet(
            in_channels=len(in_vars),  
            out_channels=len(out_vars),  
            hidden_channels=128, 
            n_blocks=28,  
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "edsr":
    net = nn.Sequential(
        EDSR(
            in_channels=len(in_vars),  
            out_channels=len(out_vars),  
            n_resblocks=28, 
            n_feats=128, 
            scale=args.ratio, 
            n_colors=1, 
            res_scale=0.1,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "ffl":
    net = nn.Sequential(
        Interpolation(args.target_size, "bilinear"),
        VisionTransformer(
            img_size=args.target_size, 
            in_channels=len(in_vars),
            out_channels=len(out_vars),
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "ffl",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "swinir":
    net = nn.Sequential(
        SwinIR(
            in_chans=len(in_vars),
            upscale=args.ratio, 
            img_size=input_size, # You need to replace to the real input size
            window_size=patch_size//2, 
            img_range=1., 
            depths=[3, 3, 3, 3],
            embed_dim=128, 
            num_heads=[4, 4, 4, 4], 
            mlp_ratio=2, 
            upsampler='pixelshuffledirect'),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "srformer":
    net = nn.Sequential(
        SRFormer(
            in_chans=len(in_vars),
            upscale=args.ratio, 
            img_size=input_size,  # You need to replace to the real input size
            window_size=patch_size//2, 
            img_range=1., 
            depths=[3, 3, 3, 3],
            embed_dim=128, 
            num_heads=[4, 4, 4, 4], 
            mlp_ratio=2, 
            upsampler='pixelshuffledirect',
            resi_connection= '1conv'
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "deepsd":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        DeepSD(
            in_channels=len(in_vars)+1,  
            out_channels=len(out_vars),  
            oro_path=ORO_PATH,
            upscale_factor=1,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "facl":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        VisionTransformer(
            img_size=target_size, 
            in_channels=len(in_vars),
            out_channels=len(out_vars),
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "facl",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        total_step=img_num//args.bs,
        micro_batch=args.bs,
    )
elif args.preset == "smcl_vit":
    net = nn.Sequential(
        Constraint_ViT(
            img_size=target_size, 
            in_channels=len(in_vars),
            out_channels=len(out_vars),
            history=1,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2, 
            num_heads=4, 
            constraints='softmax',
            upsampling_factor=args.ratio,            
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == 'dsfno':
    net = nn.Sequential(
        DSFNO(
            in_channel=len(in_vars),
            n_channels=64,
            n_residual_blocks=4,
            n_operator_blocks=2,
            modes=18,
            apply_constraint=True,
            upsample_factor=args.ratio,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        elevation=ORO_PATH,
    )
elif args.preset == "srgan":
    model_g = nn.Sequential(
        Generator(n_residual_blocks=16, 
                  upsample_factor=args.ratio, 
                  base_filter=64, 
                  num_channel=1)
    )
    model_d = nn.Sequential(
        Discriminator(base_filter=64, 
                      num_channel=1)
    )
    srgan_lr = args.lr/2
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
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=["denormalize", "denormalize", "denormalize", None],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        warmup_epochs=4,
    )
elif args.preset == "climatediffuse":
    net = nn.Sequential(
        Interpolation((534, 534), "bilinear"),
        EDMPrecond(
            img_resolution=(534, 534), #
            in_channels=2,
            out_channels=len(out_vars),
            label_dim = 2,                # Number of class labels, 0 = unconditional.
            use_fp16 = False,            # Execute the underlying model at FP16 precision?
            sigma_min = 0,                # Minimum supported noise level.
            sigma_max = float('inf'),     # Maximum supported noise level.
            sigma_data = 1.0,              # Expected standard deviation of the training data
            model_type = 'UNet',   # Class name of the underlying model.
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
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
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=["denormalize", "denormalize", "denormalize", None],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        t_hours = args.t_res,
    )
elif args.preset == 'geofar_vit':
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        GeoFAR(
            img_size=target_size, #
            in_channels=len(in_vars),
            out_channels=len(out_vars),
            history=1,
            n_coeff=64,
            n_sh_coeff=64,
            patch_size=patch_size,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2,
            num_heads=4,
            oro_path=ORO_PATH,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "resi_basis_sr",
        test_loss=["lfd","rmse", "pearson", "mean_bias"],
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        elevation=ORO_PATH
    )
elif args.preset == "geofar_unet":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        GeoFAR_Unet(
            img_size=target_size,
            in_channels=len(in_vars),  # Same as ViT
            out_channels=len(out_vars),  # Same as ViT
            history=1,
            n_coeff=64,
            n_sh_coeff=64,
            hidden_channels=64,  # Define the base number of channels
            n_blocks=2,  # U-Net depth (adjust as needed)
            ch_mults=[1, 2, 2],  # Channel multipliers for each stage
            oro_path=ORO_PATH,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)} #0.95
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "resi_basis_sr",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
    )
elif args.preset == "geocd_srgan":
    model_g = nn.Sequential(
        Interpolation((target_size, "bilinear")),
        GeoFAR_Generator(n_residual_blocks=16, 
                  upsample_factor=1, 
                  base_filter=64, 
                  num_channel=1,
                  img_size=target_size,
                  oro_path=ORO_PATH,
                  n_coeff=64, 
                  n_sh_coeff=64)
    )
    model_d = nn.Sequential(
        Discriminator(base_filter=64, 
                      num_channel=1)
    )
    srgan_lr = args.lr/2
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
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=["denormalize", "denormalize", "denormalize", None],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        warmup_epochs=4,
    )
elif args.preset == "geofar_diffuse":
    net = nn.Sequential(
        Interpolation(target_size, "bilinear"),
        EDMPrecond(
            img_resolution=target_size, 
            in_channels=len(in_vars)*2, # concatenate input and noisy image
            out_channels=len(out_vars),
            label_dim = 2,                # Number of class labels, 0 = unconditional.
            use_fp16 = False,            # Execute the underlying model at FP16 precision?
            sigma_min = 0,                # Minimum supported noise level.
            sigma_max = float('inf'),     # Maximum supported noise level.
            sigma_data = 1.0,              # Expected standard deviation of the training data
            model_type = 'GeoFAR_Diffuse',   # Class name of the underlying model.
            n_coeff = 64,
            n_sh_coeff = 64,
            oro_path = ORO_PATH,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 1e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
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
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        val_target_transform=["denormalize", "denormalize", "denormalize", None],
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        t_hours = args.t_res,
    )
elif args.preset == 'geofar_dsfno':
    net = nn.Sequential(
        GeoFAR_DSFNO(
            img_size=target_size,
            in_channel=1,
            n_channels=64,
            n_residual_blocks=4,
            n_operator_blocks=2,
            modes=18,
            apply_constraint=True,
            upsample_factor=args.ratio,
            n_coeff=64,
            n_sh_coeff=64,
            oro_path=ORO_PATH,
        ),
    )
    optim_kwargs = {"lr": args.lr, "weight_decay": 2e-4, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 10,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": 1e-6,
        "eta_min": 1e-6,
    }
    model = cl.load_downscaling_module(
        data_module=dm,
        model=net, 
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_loss= "mse",
        test_loss=["lfd","rmse", "pearson", "mean_bias"], 
        test_target_transform=["denormalize","denormalize", "denormalize", "denormalize"], 
        elevation=ORO_PATH,
    )
else:
    model = cl.load_downscaling_module(data_module=dm, architecture=args.preset) # architecture refers to predefined models in CL

# Setup trainer
current_time = datetime.datetime.now().strftime("%m%d-%H-%M")
pl.seed_everything(0)
default_root_dir = f"{args.preset}_downscaling_{args.variable}_cerra{args.ratio}_{current_time}"
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
