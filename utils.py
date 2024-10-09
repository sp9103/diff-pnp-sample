import torch
import roma
import numpy as np
from PIL import Image


l1 = lambda diff: diff.abs()
l2 = lambda diff: diff ** 2

def load_data(dir, view_id):
    image_path = f"{dir}/{view_id:06d}.png"
    depth_path = f"{dir}/{view_id:06d}_depth.png"
    intrinsic_path = f"{dir}/intrinsic.npy"
    pose_path = f"{dir}/000001.npy"
    rgba = Image.open(image_path)
    depth = Image.open(depth_path)
    intrinsic = np.load(intrinsic_path)
    pose = np.load(pose_path)[view_id]

    box = rgba.getbbox()

    rgba = np.array(rgba)
    depth = np.array(depth)
    box = np.array(box)

    # # scale
    depth = depth / 1000
    pose[:3, -1] /= 1000

    return {
        "rgba": rgba,
        "depth": depth,
        "box": box,
        "pose": pose,
        "intr": intrinsic}

def vec_to_transform(pose7d):
    '''
    7d : trans(3d) + quat(4d)
    '''
    assert pose7d.shape[-1] == 7
    bsz = pose7d.shape[0]
    trans, quats = pose7d.split([3, 4], dim=-1)
    quats = quats / torch.norm(quats, p=2, dim=-1, keepdim=True)
    R = roma.unitquat_to_rotmat(quats[..., [1,2,3,0]])
    T = torch.zeros(bsz, 4, 4, dtype=pose7d.dtype, device=pose7d.device)
    T[..., 0:3, 0:3] = R
    T[..., 0:3, 3] = trans
    T[..., 3, 3] = 1
    return T

def transform_to_vec(transform: torch.Tensor) -> torch.Tensor:
    assert transform.shape[-2:] == (4,4)
    bsz = transform.shape[0]
    quat = roma.rotmat_to_unitquat(transform[:, :3, :3])[..., [3,0,1,2]]        # wijk
    trans = transform[:, :3, 3]
    pose7d = torch.zeros(bsz, 7, dtype=transform.dtype, device=transform.device)
    pose7d[..., :3] = trans
    pose7d[..., 3:] = quat
    return pose7d

# https://github.com/cvlab-epfl/perspective-flow-aggregation/blob/0b2a7e794076f536be834853f8774b1af59daa88/flow/core/datasets.py#L209
def GetVertmapFromDepth(K, pose1, depth1):
    batch_size, height, width = depth1.shape

    # get vert map from depth1
    xs = torch.linspace(0, width-1, steps=width, device=depth1.device)
    ys = torch.linspace(0, height-1, steps=height, device=depth1.device)
    y, x = torch.meshgrid(ys, xs)
    xy1 = torch.stack([x, y, torch.ones_like(x)]).reshape(3,-1)
    xy1 = xy1.repeat(batch_size, 1, 1)
    xyn1 = torch.matmul(torch.inverse(K), xy1)

    # get 3D points in Camera coordinates
    xyzc = xyn1 * depth1.reshape(batch_size, 1, -1)
    # change to object coordinates
    vert_map = torch.matmul(pose1[...,:3].transpose(1,2), xyzc-pose1[...,3].unsqueeze(-1))
    vert_map = vert_map.reshape(batch_size, 3, height, width)

    valid_flag = (depth1 < 1e-6).unsqueeze(1).repeat(1,3,1,1)
    vert_map[valid_flag] = 0

    return vert_map, xy1

# https://github.com/cvlab-epfl/perspective-flow-aggregation/blob/0b2a7e794076f536be834853f8774b1af59daa88/flow/core/datasets.py#L231C1-L252C16
def GetFlowFromPoseAndDepth(pose1, pose2, K, depth1):
    batch_size, height, width = depth1.shape

    # get vertex map in object frame
    vert_map, xy1 = GetVertmapFromDepth(K, pose1, depth1)

    # compute new reprojections under the second pose
    R2 = pose2[..., :3]
    T2 = pose2[..., 3].unsqueeze(-1)
    xyp2 = torch.matmul(K, torch.matmul(R2, vert_map.reshape(batch_size, 3, -1)) + T2)
    x2 = xyp2[:,0] / xyp2[:,2]
    y2 = xyp2[:,1] / xyp2[:,2]

    # value to use to represent unknown flow
    UNKNOWN_FLOW = 1e10
    flowx = x2 - xy1[:,0]
    flowy = y2 - xy1[:,1]
    invalid_mask = (depth1.reshape(batch_size, -1) < 1e-6)
    flowx[invalid_mask] = UNKNOWN_FLOW
    flowy[invalid_mask] = UNKNOWN_FLOW
    flow = torch.stack([flowx, flowy]).permute(1,0,2).reshape(batch_size, 2, height, width)
    return flow

# https://github.com/ylabbe/cosypose/blob/c90a04f434b1e89f02341cc03899eb63ea8facba/cosypose/lib3d/transform_ops.py#L7
def transform_pts(T, pts):
    bsz = T.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if T.dim() == 4:
        pts = pts.unsqueeze(1)
        assert T.shape[-2:] == (4, 4)
    elif T.dim() == 3:
        assert T.shape == (bsz, 4, 4)
    else:
        raise ValueError('Unsupported shape for T', T.shape)
    pts = pts.unsqueeze(-1)
    T = T.unsqueeze(-3)
    pts_transformed = T[..., :3, :3] @ pts + T[..., :3, [-1]]
    return pts_transformed.squeeze(-1)

# https://github.com/ylabbe/cosypose/blob/c90a04f434b1e89f02341cc03899eb63ea8facba/cosypose/lib3d/cosypose_ops.py#L34
def loss_CO_symmetric(TCO_possible_gt, TCO_pred, points, l1_or_l2=l1):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert len(TCO_possible_gt.shape) == 4 and TCO_possible_gt.shape[-2:] == (4, 4)
    assert TCO_pred.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3

    TCO_points_possible_gt = transform_pts(TCO_possible_gt, points)
    TCO_pred_points = transform_pts(TCO_pred, points)
    losses_possible = l1_or_l2((TCO_pred_points.unsqueeze(1) - TCO_points_possible_gt).flatten(-2, -1)).mean(-1)
    loss, min_id = losses_possible.min(dim=1)
    TCO_assign = TCO_possible_gt[torch.arange(bsz), min_id]
    return loss, TCO_assign