"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .common import evaluate_pnp, solve_wrapper

class LMSolver(nn.Module):
    """
    Levenberg-Marquardt solver, with fixed number of iterations.

    - For 4DoF case, the pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    - For 6DoF case, the pose is parameterized as [x, y, z, w, i, j, k], where
    [w, i, j, k] is the unit quaternion.
    """
    def __init__(
            self,
            dof=4,
            num_iter=10,
            eps=1e-5):
        super(LMSolver, self).__init__()
        self.dof = dof
        self.num_iter = num_iter
        self.eps = eps

    def forward(self, x3d, x2d, w2d, camera, cost_fun,
                pose_init=None):

        pose_opt = self.solve(
            x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init)

        return pose_opt

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
            # evaluate_fun(pose, out_jacobian=None, out_residual=None, out_cost=None)
            evaluate_fun = partial(
                evaluate_pnp,
                x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun,
                clip_jac=False)

            for i in range(self.num_iter):
                residual, _, jac = evaluate_fun(pose=pose_opt, out_jacobian=True, out_residual=True)
                jac_t = jac.transpose(-1, -2)  # (num_obj, 4 or 6, num_pts * 2)
                jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)
                diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)  # (num_obj, 4 or 6)
                diagonal += self.eps  # add to jtj
                # (num_obj, 4 or 6, 1) = (num_obj, 4 or 6, num_pts * 2) @ (num_obj, num_pts * 2, 1)
                gradient = jac_t @ residual.unsqueeze(-1)
                if self.dof == 4:
                    pose_opt -= solve_wrapper(gradient, jtj).squeeze(-1)
                else:
                    step = -solve_wrapper(gradient, jtj).squeeze(-1)
                    pose_opt[..., :3] += step[..., :3]
                    pose_opt[..., 3:] = F.normalize(pose_opt[..., 3:] + (
                            camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]
                        ).squeeze(-1), dim=-1)

        else:
            pose_opt = torch.empty((0, 4 if self.dof == 4 else 7), **tensor_kwargs)

        return pose_opt
