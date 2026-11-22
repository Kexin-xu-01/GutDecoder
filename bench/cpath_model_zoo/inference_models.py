import os
from abc import abstractmethod
from typing import Optional, Tuple

import torch

import torch
from loguru import logger

import timm

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

def _get_eval_transforms(mean, std, target_img_size=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(target_img_size),
        transforms.CenterCrop(target_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _remap_layerscale(state_dict, model):
    """Bidirectionally fix LayerScale key names between old timm (<1.0, .gamma)
    and new timm (>=1.0, .weight) so checkpoints load regardless of timm version."""
    ckpt_has_gamma = any("ls1.gamma" in k for k in state_dict)
    model_has_weight = any("ls1.weight" in k for k in model.state_dict())
    if ckpt_has_gamma and model_has_weight:
        return {k.replace(".gamma", ".weight"): v for k, v in state_dict.items()}
    if not ckpt_has_gamma and not model_has_weight:
        return {
            k.replace("ls1.weight", "ls1.gamma").replace("ls2.weight", "ls2.gamma"): v
            for k, v in state_dict.items()
        }
    return state_dict


def _download_hf_timm_weights(repo_id: str) -> str:
    """Download a timm-style HF hub checkpoint and return the local path.

    Tries safetensors first, then falls back to pytorch_model.bin.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            return hf_hub_download(repo_id=repo_id, filename=fname)
        except EntryNotFoundError:
            continue
    raise RuntimeError(
        f"Could not download weights from {repo_id}. "
        "Tried model.safetensors and pytorch_model.bin."
    )
        
        
class InferenceEncoder(torch.nn.Module):
    
    def __init__(self, weights_path=None, **build_kwargs):
        super(InferenceEncoder, self).__init__()
        
        self.weights_path = weights_path
        self.model, self.eval_transforms, self.precision = self._build(weights_path, **build_kwargs)
        
    def forward(self, x):
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass
        

_TRIDENT_NEW_MODELS = {
    'kaiko-vitb16', 'kaiko-vits8', 'kaiko-vits16', 'kaiko-vitl14',
    'lunit-vits8', 'musk', 'midnight12k', 'openmidnight', 'genbio-pathfm',
    'keep',
}

def _make_trident_encoder_class(trident_name: str, precision=None):
    """
    Return an InferenceEncoder subclass that delegates to TRIDENT's encoder_factory.
    `precision` overrides the precision TRIDENT reports for the encoder.
    """
    class _Adapter(InferenceEncoder):
        def _build(self, weights_path=None):
            from trident.patch_encoder_models import encoder_factory
            enc = encoder_factory(trident_name)
            eval_transform = (
                getattr(enc, 'eval_transform', None) or
                getattr(enc, 'eval_transforms', None) or
                _get_eval_transforms(_IMAGENET_MEAN, _IMAGENET_STD)
            )
            return enc, eval_transform, precision or getattr(enc, 'precision', torch.float32)
    _Adapter.__name__ = f'TridentEncoder_{trident_name.replace("-", "_")}'
    _Adapter.__qualname__ = _Adapter.__name__
    return _Adapter


def _cls_or_cls_mean_patch(tokens, return_cls: bool, n_prefix_tokens: int):
    """
    CLS token, or CLS concatenated with the mean of the patch tokens.
    `n_prefix_tokens` is the number of CLS + register tokens before the patch tokens.
    """
    cls_token = tokens[:, 0]
    if return_cls:
        return cls_token
    return torch.cat([cls_token, tokens[:, n_prefix_tokens:].mean(1)], dim=-1)


class ConchInferenceEncoder(InferenceEncoder):
  
    def _build(self, _):

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as e:
            raise ImportError("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`") from e
        
        try:
            model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
        except Exception as e:
            raise RuntimeError("Failed to download CONCH v1 model, make sure that you were granted access and that you correctly registered your token") from e
        
        eval_transform = preprocess
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=False, normalize=False)

from trident.patch_encoder_models.load import BasePatchEncoder

# redefine. the conchv15 to take weights_path as a second argument
class Conchv15InferenceEncoder(BasePatchEncoder):

    def __init__(self, weights_path=None, **build_kwargs):
        """
        CONCHv1.5 initialization.

        HEST instantiates encoders via: EncClass(weights_path), so accept the
        first positional arg as `weights_path` and pass the rest as kwargs to BasePatchEncoder.
        """
        # store the positional weights_path for use in _build()
        self._weights_path = weights_path

        # call parent's ctor with remaining build kwargs (if it expects them)
        super().__init__(**build_kwargs)

    def _build(self, img_size=448):
        from trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained
        self.enc_name = 'conch_v15'

        self.ensure_has_internet(self.enc_name)
        try:
            model, eval_transform = create_model_from_pretrained(
                checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size
            )
        except Exception as e:
            raise RuntimeError("Failed to download CONCH v1.5 model, make sure that you were granted access and that you correctly registered your token") from e

        # set precision according to device availability (optional improvement)
        precision = torch.float16 if torch.cuda.is_available() else torch.float32

        # optionally store attributes HEST may read earlier
        self.model = model
        self.eval_transform = eval_transform
        self.precision = precision

        return model, eval_transform, precision


    
CTransPathInferenceEncoder = _make_trident_encoder_class('ctranspath', precision=torch.float32)
RemedisInferenceEncoder = _make_trident_encoder_class('remedis', precision=torch.float32)


class CustomInferenceEncoder(InferenceEncoder):
    def __init__(self, weights_path, name, model, transforms, precision):
        super().__init__(weights_path)
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision
        
    def _build(self, weights_path):
        return None, None, None
    

class PhikonInferenceEncoder(InferenceEncoder):

    def _load(self):
        from transformers import ViTModel
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        return model

    def _build(self, _, return_cls=True):
        self.return_cls = return_cls
        model = self._load()
        eval_transform = _get_eval_transforms(_IMAGENET_MEAN, _IMAGENET_STD, target_img_size=224)
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        return _cls_or_cls_mean_patch(self.forward_features(x).last_hidden_state, self.return_cls, n_prefix_tokens=1)
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out

class PhikonV2InferenceEncoder(PhikonInferenceEncoder):
    def _load(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        return model

class H0MiniInferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        _,
        return_cls=True,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        return _cls_or_cls_mean_patch(self.model(x), self.return_cls, n_prefix_tokens=5)


class ResNet50InferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        pretrained=True, 
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        pool=True
    ):
        import timm

        model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs)
        eval_transform = _get_eval_transforms(_IMAGENET_MEAN, _IMAGENET_STD)
        precision = torch.float32
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
        
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out
                     
    
class UNIInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        try:
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
        except Exception as e:
            raise RuntimeError("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token") from e
        
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        return model, eval_transform, precision


class UNI2InferenceEncoder(InferenceEncoder):
    def _build(self, weights_path=None):
        """Load MahmoodLab/UNI2-h (ViT-h/14-reg8); needs a Hugging Face token with access to the gated model."""
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        # exact timm kwargs recommended by model card
        timm_kwargs = dict(
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667*2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )

        try:
            model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        except Exception as e:
            raise RuntimeError(
                "Failed to load UNI2-h from Hugging Face. Make sure you are logged in and have accepted "
                "the model terms at https://huggingface.co/MahmoodLab/UNI2-h."
            ) from e
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        return model, eval_transform, torch.float16



class GigaPathInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        timm_kwargs={}
        ):
        import timm
        from torchvision import transforms
        
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, **timm_kwargs)

        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float32
        return model, eval_transform, precision
    
    
class VirchowInferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        _,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        precision = torch.float32
        self.return_cls = return_cls
        
        return model, eval_transform, precision
        
    def forward(self, x):
        return _cls_or_cls_mean_patch(self.model(x), self.return_cls, n_prefix_tokens=1)
        

class Virchow2InferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        _,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        return _cls_or_cls_mean_patch(self.model(x), self.return_cls, n_prefix_tokens=5)

_HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_HOPTIMUS_STD  = (0.211883, 0.230117, 0.177517)


class _HOptimusInferenceEncoder(InferenceEncoder):
    HF_REPO: str = ""

    def _build(self, weights_path=None):
        import os
        import timm
        import torch
        from torchvision import transforms

        model = timm.create_model(
            f"hf-hub:{self.HF_REPO}",
            pretrained=False,
            init_values=1e-5,
            dynamic_img_size=False,
        )

        ckpt = (
            weights_path
            if (weights_path and os.path.isfile(weights_path))
            else _download_hf_timm_weights(self.HF_REPO)
        )

        print(f"Loading H-optimus checkpoint: {ckpt}")

        if str(ckpt).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt, device="cpu")
        else:
            state_dict = torch.load(
                ckpt,
                map_location="cpu",
                weights_only=False
            )

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        state_dict = _remap_layerscale(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_HOPTIMUS_MEAN,
                std=_HOPTIMUS_STD,
            ),
        ])

        return model, eval_transform, torch.float16


class HOptimus0InferenceEncoder(_HOptimusInferenceEncoder):
    HF_REPO = "bioptimus/H-optimus-0"


class HOptimus1InferenceEncoder(_HOptimusInferenceEncoder):
    HF_REPO = "bioptimus/H-optimus-1"
         
class HibouLargeInferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        _,
        return_cls=True,
    ):
        from transformers import AutoImageProcessor, AutoModel
        _eval_transform = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
        model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)

        def eval_transform(img):
            return _eval_transform(
                img,
                return_tensors='pt'
            )['pixel_values'].squeeze()

        precision = torch.float32
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        return _cls_or_cls_mean_patch(self.model(x).last_hidden_state, self.return_cls, n_prefix_tokens=5)

class KaikoBase8InferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        _,
        return_cls=True,
    ):
        from torchvision.transforms import v2
        model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb8", trust_repo=True)
        eval_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=224),
                v2.CenterCrop(size=224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        precision = torch.float32
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        return _cls_or_cls_mean_patch(self.model.forward_features(x), self.return_cls, n_prefix_tokens=1)



class GpfmInferenceEncoder(InferenceEncoder):

    def _build(self, weights_path=None):
        import timm
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        if not weights_path or not os.path.isfile(weights_path):
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(repo_id="majiabo/GPFM", filename="GPFM.pth")

        model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=224,
            init_values=1e-5,
            num_classes=0,
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = _remap_layerscale(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

        eval_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        precision = torch.float16
        return model, eval_transform, precision


class _GemmaImageTransform:
    """Wraps Gemma4ImageProcessor as a torchvision-compatible callable."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def _load_gemma4_vision_weights(src, prefix="model.vision_tower."):
    """Parse safetensors shards and return only vision-tower weights.

    Never materialises the full 4-26B LLM in memory — reads byte ranges directly.
    Accepts either a local directory (already downloaded) or an HF repo id.
    """
    import struct
    import json as _json
    from pathlib import Path
    import numpy as np

    local_dir = Path(src) if Path(src).is_dir() else None
    if local_dir is None:
        from huggingface_hub import snapshot_download
        local_dir = Path(snapshot_download(
            src,
            ignore_patterns=["*.msgpack", "*.bin", "*.pt", "flax_*", "tf_*"],
        ))

    shards = sorted(local_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {local_dir}")

    _NP_DTYPE = {
        "F32": np.float32, "F16": np.float16, "I32": np.int32,
        "I64": np.int64, "I8": np.int8, "UI8": np.uint8,
    }
    _TORCH_DTYPE = {
        "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "I32": torch.int32, "I64": torch.int64, "I8": torch.int8, "UI8": torch.uint8,
    }

    state_dict = {}
    for shard in shards:
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = _json.loads(f.read(header_len))
            data_start = 8 + header_len
            for key, meta in header.items():
                if key == "__metadata__" or not key.startswith(prefix):
                    continue
                dtype_str = meta["dtype"]
                start, end = meta["data_offsets"]
                f.seek(data_start + start)
                raw = f.read(end - start)
                if dtype_str == "BF16":
                    tensor = torch.from_numpy(
                        np.frombuffer(raw, dtype=np.int16).copy()
                    ).view(torch.bfloat16).reshape(meta["shape"])
                else:
                    tensor = torch.from_numpy(
                        np.frombuffer(raw, dtype=_NP_DTYPE[dtype_str]).copy()
                    ).to(_TORCH_DTYPE[dtype_str]).reshape(meta["shape"])
                state_dict[key[len(prefix):]] = tensor
    if not state_dict:
        raise RuntimeError(
            f"No keys with prefix '{prefix}' found in {shards}. "
            "Check that the repo contains Gemma4ForConditionalGeneration weights."
        )
    return state_dict


class _Gemma4InferenceEncoder(InferenceEncoder):
    HF_REPO: str = ""

    def _build(self, weights_path=None):
        try:
            from transformers import Gemma4VisionModel, Gemma4ImageProcessor
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "Gemma4 encoders require transformers>=5.0 and PyTorch>=2.4.\n"
                "Install: pip install --user 'transformers>=5.0'\n"
                "Set PYTHONPATH so your user install takes priority:\n"
                "  export PYTHONPATH=~/.local/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}\n"
                "Other encoders (UNI, CONCH, GPFM, …) are NOT affected by this requirement.\n"
                f"Original error: {e}"
            ) from e

        # Use local dir only if it looks like a real HF checkpoint, not the
        # default weights_root directory returned when local_ckpts.json value is "".
        from pathlib import Path as _Path
        src = self.HF_REPO
        if weights_path:
            p = _Path(weights_path)
            if p.is_dir() and (p / "config.json").exists():
                src = weights_path

        # Memory-efficient: parse safetensors headers and load only vision tower.
        vision_state = _load_gemma4_vision_weights(src)

        # Instantiate the vision model from config (no weights), then load.
        from transformers import AutoConfig
        full_cfg = AutoConfig.from_pretrained(src)
        vision_cfg = full_cfg.vision_config
        model = Gemma4VisionModel(vision_cfg)
        missing, unexpected = model.load_state_dict(vision_state, strict=False)
        if missing:
            raise RuntimeError(
                f"Gemma4 vision state dict missing {len(missing)} key(s): {missing[:3]} …"
            )
        if unexpected:
            logger.warning(f"Gemma4 checkpoint has {len(unexpected)} key(s) the vision model does not use: {unexpected[:3]} …")

        processor = Gemma4ImageProcessor.from_pretrained(src)
        return model, _GemmaImageTransform(processor), torch.bfloat16

    def forward(self, x):
        out = self.model(pixel_values=x.to(torch.bfloat16))
        return out.last_hidden_state.mean(dim=1)


class Gemma4E4BInferenceEncoder(_Gemma4InferenceEncoder):
    HF_REPO = "google/gemma-4-E4B"


class Gemma426BInferenceEncoder(_Gemma4InferenceEncoder):
    HF_REPO = "google/gemma-4-26B-A4B"


_ENCODERS = {
    'conch_v1': ConchInferenceEncoder,
    'conch_v1_5': Conchv15InferenceEncoder,
    'uni_v1': UNIInferenceEncoder,
    'uni_v2': UNI2InferenceEncoder,
    'ctranspath': CTransPathInferenceEncoder,
    'phikon': PhikonInferenceEncoder,
    'phikon_v2': PhikonV2InferenceEncoder,
    'h0_mini': H0MiniInferenceEncoder,
    'hibou_large': HibouLargeInferenceEncoder,
    'hibou_l': HibouLargeInferenceEncoder,
    'hibou-l': HibouLargeInferenceEncoder,
    'kaiko_base_8': KaikoBase8InferenceEncoder,
    'kaiko-vitb8': KaikoBase8InferenceEncoder,
    'remedis': RemedisInferenceEncoder,
    'resnet50': ResNet50InferenceEncoder,
    'gigapath': GigaPathInferenceEncoder,
    'virchow': VirchowInferenceEncoder,
    'virchow2': Virchow2InferenceEncoder,
    'hoptimus0': HOptimus0InferenceEncoder,
    'hoptimus1': HOptimus1InferenceEncoder,
    'gpfm': GpfmInferenceEncoder,
    'gemma4-e4b': Gemma4E4BInferenceEncoder,
    'gemma4-26b': Gemma426BInferenceEncoder,
}


def inf_encoder_factory(enc_name):
    if enc_name in _ENCODERS:
        return _ENCODERS[enc_name]
    if enc_name in _TRIDENT_NEW_MODELS:
        return _make_trident_encoder_class(enc_name)
    raise ValueError(f"Unknown encoder name {enc_name}. Known: {sorted(_ENCODERS) + sorted(_TRIDENT_NEW_MODELS)}")
