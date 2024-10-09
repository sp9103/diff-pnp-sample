import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .common import evaluate_pnp, solve_wrapper

class GaussNewtonSolver(nn.Module):
    def __init__(
            self,
            dof=6,
            num_iter=10,
            eps=1e-5):
        super(GaussNewtonSolver, self).__init__()
        self.dof = dof
        self.num_iter = num_iter
        self.eps = eps

    def pose_add(self, pose_opt, step, camera):
        if self.dof == 4:
            pose_new = pose_opt + step
        else:
            pose_new = torch.cat(
                (pose_opt[..., :3] + step[..., :3],
                 F.normalize(pose_opt[..., 3:] + (
                         camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]
                     ).squeeze(-1), dim=-1)),
                dim=-1)
        return pose_new

    def gn_step(self, x3d, x2d, w2d, pose, camera, cost_fun):
        residual, _, jac = evaluate_pnp(
            x3d, x2d, w2d, pose, camera, cost_fun,
            out_jacobian=True, out_residual=True)
        jac_t = jac.transpose(-1, -2)  # (num_obj, 4 or 6, num_pts * 2)
        jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)
        jtj = jtj + torch.eye(self.dof, device=jtj.device, dtype=jtj.dtype) * self.eps
        # (num_obj, 4 or 6, 1) = (num_obj, 4 or 6, num_pts * 2) @ (num_obj, num_pts * 2, 1)
        gradient = jac_t @ residual.unsqueeze(-1)
        step = -solve_wrapper(gradient, jtj).squeeze(-1)
        return step

    def solve(self, x3d, x2d, w2d, camera, cost_fun, pose_init):
        """
        Args:
            x3d (Tensor): Shape (num_obj, num_pts, 3)
            x2d (Tensor): Shape (num_obj, num_pts, 2)
            w2d (Tensor): Shape (num_obj, num_pts, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (None | Tensor): Shape (num_obj, 4 or 7) in [x, y, z, yaw], optional

        Returns:
            tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7)
        """
        num_obj, num_pts, _ = x2d.size()
        tensor_kwargs = dict(dtype=x2d.dtype, device=x2d.device)

        if num_obj > 0:
            pose_opt = pose_init.clone()
            # iteration using GN step
            for i in range(self.num_iter):
                step = self.gn_step(x3d, x2d, w2d, pose_opt, camera, cost_fun)
                pose_opt = self.pose_add(pose_opt, step, camera)
        else:
            pose_opt = torch.empty((0, 4 if self.dof == 4 else 7), **tensor_kwargs)

        return pose_opt

    def forward(self, x3d, x2d, w2d, camera, cost_fun,
                pose_init):

        pose_opt = self.solve(
            x3d, x2d, w2d, camera, cost_fun, pose_init)

        return pose_opt
