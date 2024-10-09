import torch
import torch.nn as nn

from .camera import PerspectiveCamera
from .cost_fun import AdaptiveHuberPnPCost
from .gauss_newton import GaussNewtonSolver
from .levenberg_marquardt import LMSolver

class PoseOptimizer(nn.Module):
    def __init__(self, num_solver_steps, relative_delta=0.5):
        super(PoseOptimizer, self).__init__()
        self.num_solver_steps = num_solver_steps

        self.main_opt = LMSolver(dof=6, num_iter=num_solver_steps)
        self.sub_opt = GaussNewtonSolver(dof=6, num_iter=1)

        self.camera = PerspectiveCamera()
        self.cost_fun = AdaptiveHuberPnPCost(relative_delta=relative_delta)

    def forward(self, k, x3d, x2d, w2d, pose7d_init):
        with torch.cuda.amp.autocast(enabled=False):
            self.camera.set_param(k)
            self.cost_fun.set_param(x2d.float().detach(), w2d.float().detach())

            with torch.no_grad():
                pose = self.main_opt(
                    x3d.float(),
                    x2d.float(),
                    w2d.float(),
                    self.camera,
                    self.cost_fun,
                    pose7d_init.float()
                )

            # run sub optim
            pose = self.sub_opt(
                x3d.float(),
                x2d.float(),
                w2d,
                self.camera,
                self.cost_fun,
                pose
            )

            return pose