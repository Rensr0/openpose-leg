import gc
import cv2
import numpy as np
import math
import time
import torch
import torch.nn.functional as F

from src import util
from src.model import bodypose_model


def _enable_gpu_optimizations():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = True


def _gaussian_kernel_2d(sigma, device='cuda'):
    """生成 2D 高斯卷积核用于 GPU 上的 gaussian_filter 替代"""
    ksize = int(6 * sigma + 1) | 1  # 保证奇数
    ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def _gaussian_filter_gpu(tensor_2d, sigma, kernel_cache={}):
    """在 GPU 上对 (H, W) tensor 做高斯滤波，替代 scipy.ndimage.gaussian_filter"""
    device = tensor_2d.device
    key = (sigma, device)
    if key not in kernel_cache:
        kernel_cache[key] = _gaussian_kernel_2d(sigma, device)
    kernel = kernel_cache[key]
    ksize = kernel.shape[0]
    pad = ksize // 2

    # (H, W) → (1, 1, H, W) 做 depthwise conv
    x = tensor_2d.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, kernel_4d, padding=pad)[0, 0]


class Body(object):
    def __init__(self, model_path, boxsize=368, use_half=False):
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            _enable_gpu_optimizations()
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[GPU] 检测到 {gpu_name} ({total_mem:.1f}GB)")

        # 权重加载到 CPU → 转换 → 加载到模型 → 立即释放
        self.model = bodypose_model()
        self.model.eval()

        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model_dict = util.transfer(self.model, state_dict)
        del state_dict
        self.model.load_state_dict(model_dict)
        del model_dict
        gc.collect()

        if use_cuda:
            if use_half:
                self.model = self.model.half()
            self.model = self.model.cuda()
            torch.cuda.empty_cache()
            print(f"[GPU] 模型已加载到 GPU ({'FP16' if use_half else 'FP32'})")
        else:
            print("[Info] 运行在 CPU 模式")

        self.use_cuda = use_cuda
        self.use_half = use_cuda and use_half
        self.boxsize = boxsize

    def __call__(self, oriImg):
        scale_search = [0.5]
        stride = 8
        padValue = 128
        thre1 = 0.15
        thre2 = 0.08

        h, w = oriImg.shape[:2]
        multiplier = [x * self.boxsize / h for x in scale_search]

        heatmap_avg = None
        paf_avg = None

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)

            # numpy → GPU tensor 一步到位
            im_np = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im_np = np.ascontiguousarray(im_np)
            del imageToTest_padded

            if self.use_cuda:
                target_dtype = torch.float16 if self.use_half else torch.float32
                data = torch.from_numpy(im_np).to(device='cuda', dtype=target_dtype, non_blocking=True)
            else:
                data = torch.from_numpy(im_np)
            del im_np

            with torch.no_grad():
                out_L1, out_L2 = self.model(data)
            del data

            # ---- 全部后处理留在 GPU 上用 torch 完成 ----
            ph = imageToTest.shape[0]
            pw = imageToTest.shape[1]
            del imageToTest

            # out_L1: (1, 38, H', W'), out_L2: (1, 19, H', W')
            # squeeze + permute to (H', W', C)
            hm = out_L2[0].permute(1, 2, 0)  # (H', W', 19)
            del out_L2
            pf = out_L1[0].permute(1, 2, 0)  # (H', W', 38)
            del out_L1

            # resize stride 倍放大 + crop
            hm = hm.permute(2, 0, 1).unsqueeze(0).float()  # (1, 19, H', W')
            hm = F.interpolate(hm, scale_factor=stride, mode='bilinear', align_corners=False)
            hm = hm[0, :, :ph, :pw].permute(1, 2, 0)  # (ph, pw, 19)

            pf = pf.permute(2, 0, 1).unsqueeze(0).float()
            pf = F.interpolate(pf, scale_factor=stride, mode='bilinear', align_corners=False)
            pf = pf[0, :, :ph, :pw].permute(1, 2, 0)  # (ph, pw, 38)

            if heatmap_avg is None:
                heatmap_avg = hm / len(multiplier)
                paf_avg = pf / len(multiplier)
            else:
                heatmap_avg += hm / len(multiplier)
                paf_avg += pf / len(multiplier)
            del hm, pf

        if self.use_cuda:
            torch.cuda.empty_cache()

        # ---- 关键修复：heatmap/paf resize 回原图分辨率，保证峰值坐标精度 ----
        # 不 resize 回去的话坐标靠 scale_x/scale_y 乘法映射，会有 1-2 像素偏差
        if heatmap_avg.shape[0] != h or heatmap_avg.shape[1] != w:
            heatmap_avg = F.interpolate(
                heatmap_avg.permute(2, 0, 1).unsqueeze(0).float(),
                size=(h, w), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0)
            paf_avg = F.interpolate(
                paf_avg.permute(2, 0, 1).unsqueeze(0).float(),
                size=(h, w), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0)

        LEG_PARTS = {8, 9, 10, 11, 12, 13}

        # ---- 峰值检测全部在 GPU 上完成 ----
        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = _gaussian_filter_gpu(map_ori, sigma=2)

            # GPU 上做峰值检测
            center = one_heatmap[1:-1, 1:-1]
            peaks_binary = (
                (center >= one_heatmap[0:-2, 1:-1]) &
                (center >= one_heatmap[2:,   1:-1]) &
                (center >= one_heatmap[1:-1, 0:-2]) &
                (center >= one_heatmap[1:-1, 2:])   &
                (center > thre1)
            )
            # .nonzero() 返回 GPU tensor，转到 CPU 做坐标处理
            nonzero = peaks_binary.nonzero(as_tuple=False)  # (N, 2) on GPU → CPU
            if nonzero.numel() > 0:
                nonzero = nonzero.cpu().numpy()
                ys, xs = nonzero[:, 0], nonzero[:, 1]
            else:
                ys, xs = np.array([], dtype=np.int64), np.array([], dtype=np.int64)

            # heatmap 已经是原图分辨率，坐标直接可用
            peaks_x = xs + 1
            peaks_y = ys + 1

            if part not in LEG_PARTS:
                all_peaks.append([])
                continue

            map_ori_cpu = map_ori.cpu().numpy()
            peaks_with_score = []
            for i in range(len(peaks_x)):
                py = min(peaks_y[i], map_ori_cpu.shape[0] - 1)
                px = min(peaks_x[i], map_ori_cpu.shape[1] - 1)
                peaks_with_score.append((int(peaks_x[i]), int(peaks_y[i]), float(map_ori_cpu[py, px])))

            peaks_with_score.sort(key=lambda x: x[2], reverse=True)
            peaks_with_score = peaks_with_score[:5]

            peak_id = range(peak_counter, peak_counter + len(peaks_with_score))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks_with_score)

        # heatmap 处理完可以释放
        del heatmap_avg

        # ---- PAF 评分在 GPU 上批量完成 ----
        leg_indices = [7, 8, 10, 11]
        all_limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                       [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                       [1, 16], [16, 18], [3, 17], [6, 18]]
        all_mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                      [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                      [55, 56], [37, 38], [45, 46]]
        limbSeq = [all_limbSeq[i] for i in leg_indices]
        mapIdx = [all_mapIdx[i] for i in leg_indices]

        connection_all = []
        special_k = []
        mid_num = 8

        for k in range(len(mapIdx)):
            # PAF 通道索引 → GPU 上取
            paf_indices = [x - 19 for x in mapIdx[k]]
            score_mid = paf_avg[:, :, paf_indices]  # GPU tensor (H, W, 2)
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)

            if nA != 0 and nB != 0:
                # 批量计算所有候选对的 PAF score
                connection_candidate = []
                max_try = min(nA * nB, 50)
                count = 0

                # 预取 score_mid 到 CPU（原图分辨率）
                sm = score_mid.cpu().numpy()

                for i in range(nA):
                    for j in range(nB):
                        if count >= max_try:
                            break
                        count += 1

                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        # 坐标已经是原图分辨率，直接用
                        ax, ay = candA[i][0], candA[i][1]
                        bx, by = candB[j][0], candB[j][1]

                        startend = list(zip(np.linspace(ax, bx, num=mid_num),
                                            np.linspace(ay, by, num=mid_num)))

                        vec_x = np.array([sm[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([sm[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * h / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior,
                                 score_with_dist_prior + candA[i][2] + candB[j][2]])

                del sm
                connection_candidate.sort(key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # PAF 处理完释放
        del paf_avg

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 2 or subset[i][-2] / subset[i][-1] < 0.2:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        return candidate, subset


if __name__ == "__main__":
    import os

    model_path = '../model/body_pose_model.pth'
    test_image = '../images/20250520230644.jpg'

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
    elif not os.path.exists(test_image):
        print(f"错误：找不到图片文件 {test_image}")
    else:
        body_estimation = Body(model_path)

        oriImg = cv2.imread(test_image)
        if oriImg is None:
            print("错误：无法读取图片")
        else:
            print("开始处理...")
            start_time = time.time()
            candidate, subset = body_estimation(oriImg)
            end_time = time.time()
            print(f"处理完成！耗时: {end_time - start_time:.2f} 秒")

            import matplotlib.pyplot as plt
            canvas = util.draw_bodypose(oriImg, candidate, subset)
            plt.figure(figsize=(10, 8))
            plt.imshow(canvas[:, :, [2, 1, 0]])
            plt.axis('off')
            plt.show()
