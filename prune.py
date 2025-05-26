import os
from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect, C3k2
from torch.nn.modules.container import Sequential
from ultralytics.utils.torch_utils import get_num_params, get_flops  # ← 新增

# Uncomment and set the GPU you want to use if necessary
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class PRUNE:
    def __init__(self) -> None:
        self.threshold = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_threshold(self, model, factor: float = 0.5):
        """
        Determine a global BatchNorm weight threshold (magnitude‑based).
        Keeps the top (1‑factor) portion of channels by BN‑weight magnitude.
        """
        ws, bs = [], []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                w = m.weight.abs().detach()
                b = m.bias.abs().detach()
                ws.append(w)
                bs.append(b)
                print(
                    f"{name:<40} "
                    f"w_max={w.max():.4f} w_min={w.min():.4f} "
                    f"b_max={b.max():.4f} b_min={b.min():.4f}"
                )

        ws = torch.cat(ws)
        # Pick the weight value below which factor fraction of channels lie
        self.threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
        print(f"\nGlobal BN threshold set to {self.threshold:.6f}\n")

    def prune_conv(self, conv1: Conv, conv2):
        """
        Prune channels in conv1 (and its BN) based on the computed threshold,
        then propagate the changes to the following module(s) in conv2.
        """
        # ------------- 1. Channel selection ------------- #
        gamma = conv1.bn.weight.data.detach()
        beta = conv1.bn.bias.data.detach()

        keep_idxs = []
        local_threshold = self.threshold

        # Ensure at least 8 channels remain
        while len(keep_idxs) < 8:
            keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
            local_threshold *= 0.5

        n_keep = len(keep_idxs)
        print(f"Retaining {n_keep}/{len(gamma)} channels ({100*n_keep/len(gamma):.2f}%)")

        # ------------- 2. Prune current Conv & BN ------------- #
        conv1.bn.weight.data = gamma[keep_idxs]
        conv1.bn.bias.data = beta[keep_idxs]
        conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
        conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
        conv1.bn.num_features = n_keep

        conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
        conv1.conv.out_channels = n_keep
        if conv1.conv.bias is not None:
            conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

        # Handle Proto head (Seg models)
        if (
            isinstance(conv2, list)
            and len(conv2) > 3
            and conv2[-1]._get_name() == "Proto"
        ):
            proto = conv2.pop()
            proto.cv1.conv.in_channels = n_keep
            proto.cv1.conv.weight.data = proto.cv1.conv.weight.data[:, keep_idxs]

        # Ensure conv2 is iterable
        if not isinstance(conv2, list):
            conv2 = [conv2]

        # ------------- 3. Adjust downstream layers ------------- #
        for item in conv2:
            if item is None:
                continue

            # Plain Conv or Conv in a Sequential container
            if isinstance(item, Conv):
                conv_next = item.conv
            elif isinstance(item, Sequential):
                # Sequential(Conv, Conv) pattern
                seq_conv1 = item[0]          # first conv inside Sequential
                conv_next = item[1].conv     # following conv

                seq_conv1.conv.in_channels = n_keep
                seq_conv1.conv.out_channels = n_keep
                seq_conv1.conv.groups = n_keep
                seq_conv1.conv.weight.data = seq_conv1.conv.weight.data[keep_idxs, :]

                seq_conv1.bn.weight.data = seq_conv1.bn.weight.data[keep_idxs]
                seq_conv1.bn.bias.data = seq_conv1.bn.bias.data[keep_idxs]
                seq_conv1.bn.running_var.data = seq_conv1.bn.running_var.data[keep_idxs]
                seq_conv1.bn.running_mean.data = (
                    seq_conv1.bn.running_mean.data[keep_idxs]
                )
                seq_conv1.bn.num_features = n_keep
            else:
                conv_next = item

            # Slice input channels of following conv
            conv_next.in_channels = n_keep
            conv_next.weight.data = conv_next.weight.data[:, keep_idxs]

    def prune(self, m1, m2):
        """
        Wrapper that extracts the actual Conv modules from wrappers
        (e.g., C3k2, SPPF) and calls prune_conv.
        """
        # Unwrap first module
        if isinstance(m1, C3k2):
            m1 = m1.cv2
        if isinstance(m1, Sequential):
            # Sequential(stem_conv, main_conv)
            m1 = m1[1]

        # Ensure m2 is list‑like
        if not isinstance(m2, list):
            m2 = [m2]

        # Unwrap any C3k2/SPPF items in m2
        for idx, item in enumerate(m2):
            if isinstance(item, (C3k2, SPPF)):
                m2[idx] = item.cv1

        self.prune_conv(m1, m2)


# ---------------------------------------------------------------------- #
# Pruning pipeline
# ---------------------------------------------------------------------- #
def do_pruning(model_path: str, save_path: str):
    """
    Load a YOLO model, perform structured channel pruning, validate,
    and save the pruned checkpoint.
    """
    pruning = PRUNE()

    # 0. Load model
    yolo = YOLO(model_path)

    # 1. Compute global BN threshold (pruning ratio = 0.65)
    pruning.get_threshold(yolo.model, factor=0.1)

    # 2. Prune Bottlenecks inside each C3k2
    for _, module in yolo.model.named_modules():
        if isinstance(module, Bottleneck):
            pruning.prune_conv(module.cv1, module.cv2)

    # 3. Prune across backbone stages
    seq = yolo.model.model
    for idx in [3, 5, 7, 8]:
        pruning.prune(seq[idx], seq[idx + 1])

    # 4. Prune detection head
    detect: Detect = seq[-1]

    has_proto = hasattr(detect, "proto")  # 分割模型才有
    has_cv4 = hasattr(detect, "cv4")  # 新版 Detect 才有

    last_inputs = [seq[16], seq[19], seq[22]]
    colasts = [seq[17], seq[20], None]

    # 如果模型里没有 cv4，就用占位 None 列表补上，保持 zip 长度一致
    cv4_list = detect.cv4 if has_cv4 else [None] * len(detect.cv2)

    for idx, (last_input, colast, cv2, cv3, cv4) in enumerate(
            zip(last_inputs, colasts, detect.cv2, detect.cv3, cv4_list)):

        # 1. 组装需要一起剪枝的目标
        targets = [colast, cv2[0], cv3[0]]
        if cv4 is not None:  # 只有新版才追加
            targets.append(cv4[0])
        if idx == 0 and has_proto:  # 只有分割模型才追加 proto
            targets.append(detect.proto)

        pruning.prune(last_input, targets)

        # 2. 剪枝各 detection 分支
        pruning.prune(cv2[0], cv2[1])
        pruning.prune(cv2[1], cv2[2])

        pruning.prune(cv3[0], cv3[1])
        pruning.prune(cv3[1], cv3[2])

        if cv4 is not None:  # 如果存在 cv4，再剪 dfl 分支
            pruning.prune(cv4[0], cv4[1])
            pruning.prune(cv4[1], cv4[2])

    img_size = 640  # 计算 FLOPs 时用的输入分辨率
    params_m = get_num_params(yolo.model) / 1e6  # → M
    flops_g = get_flops(yolo.model, imgsz=img_size)  # → G

    print(f"\nPruned model summary @ {img_size}×{img_size}: "
          f"{params_m:.2f}M params | {flops_g:.2f}GFLOPs\n")

    # 5. Re‑enable grads (if you plan to fine‑tune) and save
    for _, p in yolo.model.named_parameters():
        p.requires_grad = False

    # Quick validation (adjust args as needed)
    #yolo.val(data="/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/datasets/myVisDrone.yaml", batch=8, device="1", workers=2)

    # Save pruned weights
    #torch.save(yolo.ckpt, save_path)
    #print(f"Pruned checkpoint saved to: {save_path}")


if __name__ == "__main__":
    model_path = "/home/hnu3/mnt/yyk/yolov11/yolo11n-visdrone.pt"
    save_path = "/home/hnu3/mnt/yyk/yolov11/yolo11n-prune-visdrone.pt"
    do_pruning(model_path, save_path)
