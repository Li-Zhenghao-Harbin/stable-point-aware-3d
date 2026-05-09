# Set HF hub timeouts before huggingface_hub (pulled in via spar3d.system) reads constants.
import os
import sys

# huggingface_hub defaults HF_HUB_DOWNLOAD_TIMEOUT to 10s — too low for multi-GB weights on slow links.
if os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") is None:
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

import argparse
from contextlib import nullcontext

import requests
import torch
from huggingface_hub.utils import GatedRepoError
from PIL import Image
from tqdm import tqdm
from transparent_background import Remover

from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, get_device, remove_background


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", type=str, nargs="+", help="Path to input image(s) or folder."
    )
    parser.add_argument(
        "--device",
        default=get_device(),
        type=str,
        help=(
            "Device: cuda | cuda:0 | mps | cpu. Default follows torch (cuda if available). "
            "Use --device cuda to require GPU; install PyTorch with CUDA from pytorch.org if you see CPU only."
        ),
    )
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-point-aware-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/stable-point-aware-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=1.3,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="Output directory to save the results. Default: 'output/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Texture atlas resolution. Default: 1024",
    )
    parser.add_argument(
        "--low-vram-mode",
        action="store_true",
        help=(
            "Use low VRAM mode. SPAR3D consumes 10.5GB of VRAM by default. "
            "This mode will reduce the VRAM consumption to roughly 7GB but in exchange "
            "the model will be slower. Default: False"
        ),
    )

    remesh_choices = ["none"]
    if TRIANGLE_REMESH_AVAILABLE:
        remesh_choices.append("triangle")
    if QUAD_REMESH_AVAILABLE:
        remesh_choices.append("quad")
    parser.add_argument(
        "--remesh_option",
        choices=remesh_choices,
        default="none",
        help="Remeshing option",
    )
    if TRIANGLE_REMESH_AVAILABLE or QUAD_REMESH_AVAILABLE:
        parser.add_argument(
            "--reduction_count_type",
            choices=["keep", "vertex", "faces"],
            default="keep",
            help="Vertex count type",
        )
        parser.add_argument(
            "--target_count",
            type=check_positive,
            help="Selected target count.",
            default=2000,
        )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    args = parser.parse_args()

    requested = args.device
    use_cuda = requested == "cuda" or requested.startswith("cuda:")
    use_mps = requested == "mps"
    if not (use_cuda or use_mps or requested == "cpu"):
        raise ValueError("Invalid device. Use cuda, cuda:N, mps or cpu")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if use_cuda:
        if not torch.cuda.is_available():
            print(
                "\n[设备] 使用了 --device cuda，但 torch.cuda.is_available() 为 False，无法使用 GPU。\n"
                "  常见原因：当前环境是 CPU 版 PyTorch。请先检查：\n"
                '    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"\n'
                "  在本仓库根目录可一键重装带 CUDA 的 torch（默认 cu124，可按需改 requirements-cuda.txt 里的索引）：\n"
                "    pip install -r requirements-cuda.txt\n"
                "  或按官网自行选择 CUDA 版本：\n"
                "    https://pytorch.org/get-started/locally/\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
        device = requested
    elif use_mps:
        if not torch.backends.mps.is_available():
            print(
                "\n[设备] 指定了 mps，但当前 PyTorch/系统不支持 MPS。\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
        device = "mps"
    elif requested == "cpu":
        device = "cpu"
    else:
        raise ValueError("Invalid device. Use cuda, cuda:N, mps or cpu")

    print("Device used:", device)
    if device == "cpu":
        print(
            "[提示] 正在使用 CPU。若本机有 NVIDIA GPU，请安装 CUDA 版 PyTorch 后用 --device cuda。",
            file=sys.stderr,
        )

    try:
        model = SPAR3D.from_pretrained(
            args.pretrained_model,
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=args.low_vram_mode,
        )
    except GatedRepoError as err:
        print(
            "\n[Hugging Face] 无法下载模型：该仓库为 gated，需要先在网页同意条款并拥有访问权限，"
            "再用有效 token 登录。\n"
            "  1) 打开并申请访问: https://huggingface.co/stabilityai/stable-point-aware-3d\n"
            "  2) 创建 read token: https://huggingface.co/settings/tokens\n"
            "  3) 终端执行: huggingface-cli login\n"
            "     或 (PowerShell) 设置: $env:HF_TOKEN = \"<你的token>\"\n"
            "  若模型已克隆到本机目录，可加: --pretrained-model <本地路径>\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from err
    except (requests.exceptions.RequestException, OSError) as err:
        err_s = str(err).lower()
        if any(
            x in err_s or x in type(err).__name__.lower()
            for x in ("timeout", "incompleteread", "connection", "chunked", "ssl")
        ):
            print(
                "\n[Hugging Face] 下载权重失败（网络超时或中断）。model.safetensors 约 7GB，弱网容易断。\n"
                "  可加大超时后重试（PowerShell）：\n"
                '    $env:HF_HUB_DOWNLOAD_TIMEOUT = "3600"\n'
                "  或使用官方 CLI 断点续传到缓存后再运行：\n"
                "    huggingface-cli download stabilityai/stable-point-aware-3d model.safetensors config.yaml\n"
                "  若已有人工下载的目录，使用: --pretrained-model <含 config.yaml 与 model.safetensors 的文件夹>\n",
                file=sys.stderr,
            )
        raise SystemExit(1) from err
    model.to(device)
    model.eval()

    bg_remover = Remover(device=device)
    images = []
    idx = 0
    for image_path in args.image:

        def handle_image(image_path, idx):
            image = remove_background(
                Image.open(image_path).convert("RGBA"), bg_remover
            )
            image = foreground_crop(image, args.foreground_ratio)
            os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
            image.save(os.path.join(output_dir, str(idx), "input.png"))
            images.append(image)

        if os.path.isdir(image_path):
            image_paths = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            for image_path in image_paths:
                handle_image(image_path, idx)
                idx += 1
        else:
            handle_image(image_path, idx)
            idx += 1

    # reduction_count_type / target_count exist only when gpytoolbox or pynanoinstantmeshes is installed.
    reduction_type = getattr(args, "reduction_count_type", "keep")
    target_cnt = getattr(args, "target_count", 2000)
    vertex_count = (
        -1
        if reduction_type == "keep"
        else (target_cnt if reduction_type == "vertex" else target_cnt // 2)
    )

    for i in tqdm(range(0, len(images), args.batch_size)):
        image = images[i : i + args.batch_size]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with (
                torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else device,
                    dtype=torch.bfloat16,
                )
                if device.startswith("cuda")
                else nullcontext()
            ):
                mesh, glob_dict = model.run_image(
                    image,
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                    vertex_count=vertex_count,
                    return_points=True,
                )
        if torch.cuda.is_available():
            print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
        elif torch.backends.mps.is_available():
            print(
                "Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB"
            )

        if len(image) == 1:
            out_mesh_path = os.path.join(output_dir, str(i), "mesh.glb")
            mesh.export(out_mesh_path, include_normals=True)
            out_points_path = os.path.join(output_dir, str(i), "points.ply")
            glob_dict["point_clouds"][0].export(out_points_path)
        else:
            for j in range(len(mesh)):
                out_mesh_path = os.path.join(output_dir, str(i + j), "mesh.glb")
                mesh[j].export(out_mesh_path, include_normals=True)
                out_points_path = os.path.join(output_dir, str(i + j), "points.ply")
                glob_dict["point_clouds"][j].export(out_points_path)
