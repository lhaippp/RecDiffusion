import inspect

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb


def load_flow(path):
    if path.endswith(".png"):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flo_file = cv2.imread(path, -1)
        flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
        invalid = flo_file[:, :, 0] == 0  # mask
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return flo_img, np.expand_dims(flo_file[:, :, 0], 2)
    else:
        with open(path, "rb") as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert 202021.25 == magic, "Magic number incorrect. Invalid .flo file"
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D


def flow_warp(x, flow12, pad="border", mode="bilinear"):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if "align_corners" in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def flow_to_vgrid(flow):
    B, _, H, W = flow.size()
    base_grid = mesh_grid(B, H, W).type_as(flow)
    """ print(torch.max(base_grid))
    print(torch.min(base_grid)) """
    v_grid = norm_grid(base_grid + flow)
    """ print(torch.max(v_grid))
    print(torch.min(v_grid)) """
    return v_grid.permute(0, 3, 1, 2)


def vgrid_to_flow(vgrid):
    B, _, H, W = vgrid.size()
    base_grid = mesh_grid(B, H, W).type_as(vgrid)

    flow = unnorm_grid(vgrid).permute(0, 3, 1, 2) - base_grid
    return flow


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()
    """ print(torch.max(v_grid))
    print(torch.min(v_grid)) """
    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    """ print(torch.max(v_grid_norm))
    print(torch.min(v_grid_norm)) """
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def unnorm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = (v_grid[:, 0, :, :] + 1) * (W - 1) / 2
    v_grid_norm[:, 1, :, :] = (v_grid[:, 1, :, :] + 1) * (H - 1) / 2
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # 找到float pixel的最近四个int pixel
    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    # 判断是越界
    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor

    invalid = torch.cat([x_ceil_out | y_ceil_out, x_ceil_out | y_floor_out, x_floor_out | y_ceil_out, x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W, x_ceil + y_floor * W, x_floor + y_ceil * W, x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)

    values = torch.cat(
        [
            (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
            (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
            (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
            (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor)),
        ],
        1,
    )

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)
    return corresponding_map.unsqueeze(1)


def get_occu_mask_backward(flow, th=0.2):
    b, c, h, w = flow.size()
    base_grid = mesh_grid(b, h, w).type_as(flow)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow)  # BHW
    occu_mask = corr_map.clamp(min=0.0, max=1.0) < th
    return occu_mask.float()


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad="zeros")
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def flow_to_image(flow, max_flow=256):
    # flow shape (H, W, C)
    if max_flow is not None:
        max_flow = max(max_flow, 1.0)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


def visulize_flow(output_batch):
    flows_fw = output_batch["flow_fw"][0]

    vis_flow = []
    np_flow = flows_fw.detach().cpu().numpy().transpose([0, 2, 3, 1])

    for _, flow in enumerate(np_flow):
        vis_flow.append(flow_to_image(flow))
    vis_flow_np = np.array(vis_flow, dtype=np.uint8)
    vis_flow_np = vis_flow_np.transpose([0, 3, 1, 2])
    # print(vis_flow_np.shape)
    return vis_flow_np


def convert_flow_np(output_batch):
    flows_fw = output_batch["flow_fw"][0]
    np_flow = flows_fw.detach().cpu().numpy().transpose([0, 2, 3, 1])
    return np_flow


def flow_to_image_luo(flow, display=False):
    """

    :param flow: H,W,2
    :param display:
    :return: H,W,3
    """

    def compute_color(u, v):
        def make_color_wheel():
            """
            Generate color wheel according Middlebury color code
            :return: Color wheel
            """
            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR

            colorwheel = np.zeros([ncols, 3])

            col = 0

            # RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
            col += RY

            # YG
            colorwheel[col : col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
            colorwheel[col : col + YG, 1] = 255
            col += YG

            # GC
            colorwheel[col : col + GC, 1] = 255
            colorwheel[col : col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
            col += GC

            # CB
            colorwheel[col : col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
            colorwheel[col : col + CB, 2] = 255
            col += CB

            # BM
            colorwheel[col : col + BM, 2] = 255
            colorwheel[col : col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
            col += +BM

            # MR
            colorwheel[col : col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
            colorwheel[col : col + MR, 0] = 255

            return colorwheel

        """
            compute optical flow color map
            :param u: optical flow horizontal map
            :param v: optical flow vertical map
            :return: optical flow in color code
            """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2 + v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    UNKNOWN_FLOW_THRESH = 1e7
    """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    _min, _mean, _max = np.min(flow), np.mean(flow), np.max(flow)

    return np.uint8(img), (_min, _mean, _max)


def im_norm(img):
    # a = np.max(img)
    # b = np.min(img)
    # c = np.mean(img)
    # print("before clip max: {} - min: {} - mean: {}".format(a, b, c))
    # img = np.clip(img, 1.5 * c, a)

    a = np.max(img)
    b = np.min(img)
    c = np.mean(img)
    print("after clip max: {} - min: {} - mean: {}".format(a, b, c))
    img = (img - b) / (a - b)
    img = img * 255
    img = np.clip(img, 0, 200)
    img = img.astype("uint8")
    return img


def im_std_norm(img):
    a = np.max(img)
    b = np.min(img)
    c = np.mean(img)
    print("max: {} - min: {} - mean: {}".format(a, b, c))
    img = np.clip(img, 0, 10 * c)

    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    img = img * 255
    img = img.astype("uint8")
    return img


def euclidean(t):
    return torch.sqrt(torch.sum(t**2, dim=(1,), keepdim=True))


def flow_error_avg(pred_flow, gt_flow):
    B, _, H, W = gt_flow.shape
    _, _, h, w = pred_flow.shape
    assert (H == h) and (W == w), "inps shape {} is not the same as target shape {}".format(pred_flow.shape, gt_flow.shape)

    diff = euclidean(pred_flow - gt_flow)
    # diff = torch.norm(pred_flow - gt_flow, p=2, dim=1)

    PCK_1 = diff.le(1.0).float().mean()
    PCK_5 = diff.le(5.0).float().mean()
    diff_s = torch.mean(diff)

    error = diff_s
    return error, PCK_1, PCK_5


def mesh_grid_np(B, H, W):
    # mesh grid
    x_base = np.arange(0, W)
    x_base = np.tile(x_base, (B, H, 1))

    y_base = np.arange(0, H)  # BHW
    y_base = np.tile(y_base, (B, W, 1)).transpose(0, 2, 1)

    ones = np.ones_like(x_base)

    base_grid = np.stack([x_base, y_base, ones], 1)  # B3HW
    return base_grid


def get_flow_np(H_mat_mul, patch_indices, image_size_h=600, image_size_w=800):
    # (N, 6, 3, 3)
    batch_size = H_mat_mul.shape[0]
    divide = H_mat_mul.shape[1]
    H_mat_mul = H_mat_mul.reshape(batch_size, divide, 3, 3)

    small_patch_sz = [image_size_h // divide, image_size_w]

    H_mat_pool = np.zeros((batch_size, image_size_h, image_size_w, 3, 3))

    for i in range(divide):
        H_mat = H_mat_mul[:, i, :, :]

        if i == divide - 1:
            H_mat = np.broadcast_to(np.expand_dims(np.expand_dims(H_mat, 1), 1), (batch_size, image_size_h - i * small_patch_sz[0], image_size_w, 3, 3))
            H_mat_pool[:, i * small_patch_sz[0] :, ...] = H_mat
            continue

        H_mat = np.broadcast_to(np.expand_dims(np.expand_dims(H_mat, 1), 1), (batch_size, small_patch_sz[0], image_size_w, 3, 3))
        H_mat_pool[:, i * small_patch_sz[0] : (i + 1) * small_patch_sz[0], ...] = H_mat

    pred_I2_index_warp = np.expand_dims(patch_indices.transpose(0, 2, 3, 1), 4)
    pred_I2_index_warp = np.matmul(H_mat_pool, pred_I2_index_warp)[:, :, :, :, 0].transpose(0, 3, 1, 2)
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6
    T_t = T_t + smallers
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    warp_index = np.concatenate((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = warp_index - vgrid
    # NCHW to HWC
    return flow.squeeze().transpose(1, 2, 0)


def homo_to_flow(homo, H=600, W=800):
    img_indices = mesh_grid_np(B=1, H=H, W=W)
    flow_gyro = get_flow_np(homo, img_indices, image_size_h=H, image_size_w=W)
    return flow_gyro.astype(np.float32)


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        u_scale = w / w_
        v_scale = h / h_
        u, v = res.chunk(2, dim=1)
        u = u * u_scale
        v = v * v_scale
        res = torch.cat([u, v], dim=1)
    return res


def norm_flow(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = v_grid[:, 0, :, :] / (W - 1)
    v_grid_norm[:, 1, :, :] = v_grid[:, 1, :, :] / (H - 1)
    return v_grid_norm  # BHW2


def unnorm_flow(v_grid_norm):
    _, _, H, W = v_grid_norm.size()

    # scale grid to [-1,1]
    v_grid = torch.zeros_like(v_grid_norm)

    v_grid[:, 0, :, :] = (W - 1) * v_grid_norm[:, 0, :, :]
    v_grid[:, 1, :, :] = (H - 1) * v_grid_norm[:, 1, :, :]
    return v_grid  # BHW2


def flow_mask(flow):
    b, c, h, w = flow.size()
    ones_image = torch.ones([b, 3, h, w]).to(flow.device)
    mask = flow_warp(ones_image, flow, pad="zeros", mode="bilinear")
    mask = torch.where(mask > 0, torch.tensor(1), mask)
    return mask
