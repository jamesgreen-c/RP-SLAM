import jax.numpy as jnp

from src import distmaps
from src.rp_slam import distributions
from src.rp_slam.recognition import rpm
from src.rp_slam.free_energy import rpm_fe
from src.rp_slam.config import Config
from src.rp_slam.free_energy import ramp_fe
from src.rp_slam.recognition import ramp
from src import networks


def setup(
    kernel: str,
    latent_dim: int,
    batch_size: int,
    num_iter: int,
    seed: int,
    prior_lr: float = 1e-4,
    rec_lr: float = 1e-4
):
    D = latent_dim

    CFG = Config(
        num_iter=num_iter,
        batch_size=batch_size,
        jit=True,
        seed=seed,
    )

    # learning rates
    CFG.prior.lr = prior_lr
    CFG.recognition.lr = rec_lr

    REC = rpm.GaussianRecognition(
        network=networks.MLP([256, 256, 256]),
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

    else:
        CFG.stabilise_A = "clip"
        
        # Match the existing RPSSM prior initialisation.
        PRIOR = distributions.LGStationaryParam(
            stationary=True, 
            A=0.5*jnp.eye(D),
        )
        MODEL = rpm.RPSSM(prior=PRIOR, recognition=(REC,))
        MODEL_FE = rpm_fe.ConstrainedIVFreeEnergy(model=MODEL)


    return CFG, MODEL_FE

