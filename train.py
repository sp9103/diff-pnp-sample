import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import trimesh
import cv2

from utils import *
from pnp.pose_optimizer import PoseOptimizer
from flow_viz import flow_to_image


torch.manual_seed(241010)

class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = 3

        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
        )
    
    def forward(self, left, right):
        input = torch.concat([left, right], dim=1)
        output = self.net(input)
        
        flow = output[:, :2]
        weight = output[:, 2:].sigmoid()

        return flow, weight
    
def pose(TCO_input, intr_batch, left_depths, flow, weight, pose_optim, coords):
    pose7d_init = transform_to_vec(TCO_input)
    vert_map, _ = GetVertmapFromDepth(intr_batch, TCO_input[:, :3], left_depths)

    left_mask = (left_depths[:, None] > 1e-6)

    x3d = vert_map.flatten(2).transpose(-1,-2) # (b, f_h*f_w, 3)
    x2d = (coords + flow).flatten(2).transpose(-1,-2) # (b, f_h*f_w, 2)
    w2d = (weight * left_mask).flatten(2).transpose(-1, -2) # (b, f_h*f_w, 1)

    # pose
    pose_opt = pose_optim(intr_batch, x3d, x2d, w2d, pose7d_init)

    return pose_opt


if __name__ == "__main__":
    epoch = 1000
    batch = 32

    left_data = load_data('data', 11)
    right_data = load_data('data', 12)

    model = FlowModel().cuda()
    pose_optim = PoseOptimizer(3).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    left_batch = torch.from_numpy(np.stack([left_data['rgba'][..., :3]] * batch)).permute(0,3,1,2).cuda() / 255
    right_batch = torch.from_numpy(np.stack([right_data['rgba'][..., :3]] * batch)).permute(0,3,1,2).cuda() / 255
    intr_batch = torch.from_numpy(np.stack([right_data['intr'][0]] * batch)).cuda().float()
    TCO_input = torch.from_numpy(np.stack([left_data['pose']] * batch)).cuda()
    TCO_gt = torch.from_numpy(np.stack([right_data['pose']] * batch)).cuda()
    left_depths = torch.from_numpy(np.stack([left_data['depth']] * batch)).cuda().float()

    mesh = trimesh.load('data/obj_000001.ply')
    sampled_points = torch.from_numpy(np.stack([mesh.sample(1000, return_index=False)] * batch)).cuda().float() / 1000

    _, _, h, w = left_batch.shape

    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords =  coords[None].repeat(batch, 1, 1, 1).cuda()

    for epoch_id in range(epoch):
        flow, weight = model(left_batch, right_batch)   # left -> right flow

        pose_opt = pose(TCO_input, intr_batch, left_depths, flow, weight, pose_optim, coords)
        TCO_refined = vec_to_transform(pose_opt)

        # loss
        pose_loss, _ = loss_CO_symmetric(
            TCO_gt.unsqueeze(1),
            TCO_refined,
            sampled_points,
        )
        pose_loss = pose_loss.mean()
        left_mask = (left_depths[:, None] > 1e-6)
        loss = pose_loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pose_gt = transform_to_vec(TCO_gt)
        dist_t = (pose_opt[:, :3] - pose_gt[:, :3]).norm(dim=-1)
        dot_quat = (pose_opt[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
        dist_theta = 2 * torch.acos(dot_quat.abs())

        print(f'Epoch {epoch_id + 1}: loss={loss:.4f}, pose loss={pose_loss:.4f}, mean_trans_error: {dist_t.mean():.4f}, mean_ori_error: {dist_theta.mean():.4f}')

    # evaluation
    with torch.inference_mode():
        model.eval()
        flow, weight = model(left_batch[[0]], right_batch[[0]])
        pose_opt = pose(TCO_input[[0]], intr_batch[[0]], left_depths[[0]], flow, weight, pose_optim, coords[[0]])
        TCO_refined = vec_to_transform(pose_opt)

        print('===TCO Init===')
        print(TCO_input[0])
        print('===TCOrefined===')
        print(TCO_refined[0])
        print('===TCO gt===')
        print(TCO_gt[0])

        pose_gt = transform_to_vec(TCO_gt[[0]])
        pose_init = transform_to_vec(TCO_input[[0]])
        print('====before====')
        dist_t = (pose_init[:, :3] - pose_gt[:, :3]).norm(dim=-1)
        dot_quat = (pose_init[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
        dist_theta = 2 * torch.acos(dot_quat.abs())

        print(f'trans error : {dist_t.mean():.4f}')
        print(f'ori error : {dist_theta.mean():.4f}')
        print('====after====')
        dist_t = (pose_opt[:, :3] - pose_gt[:, :3]).norm(dim=-1)
        dot_quat = (pose_opt[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
        dist_theta = 2 * torch.acos(dot_quat.abs())

        print(f'trans error : {dist_t.mean():.4f}')
        print(f'ori error : {dist_theta.mean():.4f}')

        flow_est_viz = flow_to_image(flow.permute(0,2,3,1)[0].cpu().numpy(), convert_to_bgr=True)
        weight_viz = (weight * left_mask[[0]])[0,0]
        weight_viz = (weight_viz - weight_viz.min()) / (weight_viz.max() - weight_viz.min())
        weight_viz = np.stack([(weight_viz.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)

        cv2.imwrite('flow.jpg', np.hstack([flow_est_viz, weight_viz]))