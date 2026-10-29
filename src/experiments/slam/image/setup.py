import jax.numpy as jnp

from src import distmaps, distributions, networks
from src.rp_slam.config import Config as RPConfig
from src.rp_slam.free_energy import rpm_fe, ramp_fe
from src.rp_slam.recognition import rpm, ramp
from src.svae_slam.config import Config as SVAEConfig
from src.svae_slam import svae_fe, svae

def setup(
    kernel: str,
    latent_dim: int,
    batch_size: int,
    num_iter: int,
    seed: int,
    prior_lr: float = 1e-4,
    rec_lr: float = 1e-4,
    dec_lr: float = 1e-4,
    obs_dim: int = None
):
    if kernel not in {"ramp", "rpssm", "svae"}:
        raise NotImplementedError(f"{kernel} not implemented in setup.py")

    D = latent_dim
    Config = SVAEConfig if kernel == "svae" else RPConfig

    CFG = Config(
        num_iter=num_iter,
        batch_size=batch_size,
        jit=True,
        seed=seed,
    )
    CFG.prior.lr = prior_lr
    CFG.recognition.lr = rec_lr

    REC = rpm.GaussianRecognition(
        network=networks.CNN([{"features": 64, "kernel_size": 5},
                              {"features": 128, "kernel_size": 3},
                              {"features": 256, "kernel_size": 3}]),
        dist_map=distmaps.MVNCholeskySoftplus(D),
        constant_cov=True,
    )

    # model definition
    if kernel == "ramp":
        RPM = rpm.GaussianRecognition(
            network=networks.TanhMLP([128]),
            dist_map=distmaps.MVNCholeskySoftplus(D),
            constant_cov=True
        )
        MODEL = ramp.RAMPSSM(
            latent_dim=D,
            recognition=(REC,),
            message_network=RPM,
        )
        MODEL_FE = ramp_fe.ConstrainedIVFreeEnergy(model=MODEL)

    elif kernel == "rpssm":
        CFG.stabilise_A = "clip"
        
        # Match the existing RPSSM prior initialisation.
        PRIOR = distributions.LGStationaryParam(
            stationary=True, 
            A=0.5*jnp.eye(D),
        )
        MODEL = rpm.RPSSM(prior=PRIOR, recognition=(REC,))
        MODEL_FE = rpm_fe.ConstrainedIVFreeEnergy(model=MODEL)

    else:
        CFG.stabilise_A = "clip"
        CFG.generation.lr = dec_lr

        DEC = rpm.GaussianRecognition(
            network=networks.MLP([256, 256, 256]),
            dist_map=distmaps.MVNDiagVector(obs_dim),
            constant_cov=True,
        )
        PRIOR = distributions.LGStationaryParam(
            stationary=True,
            A=0.5 * jnp.eye(D),
        )
        MODEL = svae.SVAE(
            prior=PRIOR,
            recognition=(REC,),
            generation=(DEC,),
        )
        MODEL_FE = svae_fe.FreeEnergy(model=MODEL)
    return CFG, MODEL_FE

