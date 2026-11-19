"""Original Base versus target relocated before first observation; fresh inference.

Both variants use native settings and the original 520-step budget.
Videos are 20 Hz simulation time.
"""
import collections
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import copy

OUT = Path(__file__).resolve().parent
ROOT = Path('${XVLA_ROOT}')
OLD = ROOT / "libero_max_lite_20261115"
REPO = ROOT / "vendor/LIBERO-MAX"
sys.path[:0] = [str(OLD), str(REPO / "scripts/libero_source_overlay"),
                str(REPO / "src"), str(REPO / "scripts"), str(ROOT / "vendor/lerobot/src")]
import compatibility
compatibility.install()
from runner_local import _create_libero_env, _predict_action_chunk
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero_max.cosmos_integration import CosmosInterventionEnv
from libero_max.env_factory import create_libero_env_with_retry
from libero_max.substrate import load_case_task
from libero_max.pro_runtime import wrap_case_env
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.libero import get_libero_dummy_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors
from lerobot.utils.constants import ACTION

CASE_ID = "libero_10-t0235-target_relocation-d1-p195"
case = next(c for c in json.loads((OLD / "video_cases.json").read_text())["cases"] if c["case_id"] == CASE_ID)
original = json.loads((OLD / "plus/cases" / CASE_ID / "paired_summary.json").read_text())
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
def reseed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def frame(obs, name, step, phase):
    canvas = Image.new('RGB', (720,420), (18,22,30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10,5), f'{name} | Step {step}/520', font=font, fill='white')
    draw.text((10,31), phase + ' | simulation 20 Hz', font=font, fill=(255,200,90))
    for i,key in enumerate(('agentview_image','robot0_eye_in_hand_image')):
        canvas.paste(Image.fromarray(np.ascontiguousarray(obs[key][::-1])), (360*i,60))
    return np.array(canvas)

