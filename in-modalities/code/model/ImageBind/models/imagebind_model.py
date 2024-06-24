import os
import urllib
from functools import partial
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn

from .helpers import (
    EinOpsRearrange,
    LearnableLogitScaling,
    Normalize,
    SelectElement,
    SelectEOSAndProject,
)
from .multimodal_preprocessors import (
    AudioPreprocessor,
    IMUPreprocessor,
    PadIm2Video,
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    TextPreprocessor,
    ThermalPreprocessor,
)

from .transformer import MultiheadAttention, SimpleTransformer


ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


@dataclass
class EncoderArgs():
    video_frames=2
    kernel_size=(2, 14, 14)
    audio_kernel_size=16
    audio_stride=10
    out_embed_dim=1024
    vision_embed_dim=1280
    vision_num_blocks=32
    vision_num_heads=16
    audio_embed_dim=768
    audio_num_blocks=12
    audio_num_heads=12
    audio_num_mel_bins=128
    audio_target_len=204
    audio_drop_path=0.1
    text_embed_dim=1024
    text_num_blocks=24
    text_num_heads=16
    depth_embed_dim=384
    depth_kernel_size=16
    depth_num_blocks=12
    depth_num_heads=8
    depth_drop_path=0.0
    thermal_embed_dim=768
    thermal_kernel_size=16
    thermal_num_blocks=12
    thermal_num_heads=12
    thermal_drop_path=0.0
    imu_embed_dim=512
    imu_kernel_size=8
    imu_num_blocks=6
    imu_num_heads=8
    imu_drop_path=0.7


class ImageBindModel(nn.Module):
    def __init__(self, args: EncoderArgs):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(args)

        self.modality_trunks = self._create_modality_trunks(args)

        self.modality_heads = self._create_modality_heads(args)

        self.modality_postprocessors = self._create_modality_postprocessors(args)

    def _create_modality_preprocessors(self, args: EncoderArgs):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=args.kernel_size,
                    out_channels=args.vision_embed_dim,
                    stride=args.kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, args.video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=args.text_embed_dim,
            causal_masking=True,
        )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=args.audio_kernel_size,
                    stride=args.audio_stride,
                    out_channels=args.audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=args.audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, args.audio_num_mel_bins, args.audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=args.depth_kernel_size,
                    in_channels=1,
                    out_channels=args.depth_embed_dim,
                    stride=args.depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=args.depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=args.thermal_kernel_size,
                    in_channels=1,
                    out_channels=args.thermal_embed_dim,
                    stride=args.thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=args.thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=args.imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=args.imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=args.imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=args.imu_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(self, args: EncoderArgs):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            args.vision_embed_dim,
            args.vision_num_blocks,
            args.vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            args.text_embed_dim,
            args.text_num_blocks,
            args.text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            args.audio_embed_dim,
            args.audio_num_blocks,
            args.audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=args.audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            args.depth_embed_dim,
            args.depth_num_blocks,
            args.depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=args.depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            args.thermal_embed_dim,
            args.thermal_num_blocks,
            args.thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=args.thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            args.imu_embed_dim,
            args.imu_num_blocks,
            args.imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=args.imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(self, args: EncoderArgs):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=args.vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(args.vision_embed_dim, args.out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=args.text_embed_dim, eps=1e-6),
                nn.Linear(args.text_embed_dim, args.out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=args.audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(args.audio_embed_dim, args.out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=args.depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(args.depth_embed_dim, args.out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=args.thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(args.thermal_embed_dim, args.out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=args.imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(args.imu_embed_dim, args.out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}
        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(learnable=True))
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=20.0, learnable=False))
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=5.0, learnable=False))
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=10.0, learnable=False))
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=5.0, learnable=False))
        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (modality_value.ndim >= 5)  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(B * S, *modality_value.shape[2:])

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](**{modality_key: modality_value})
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](modality_value, **head_inputs)
                if modality_key in [ModalityType.AUDIO]:
                    modality_value = self.modality_postprocessors[modality_key][0](modality_value)
                else:
                    modality_value = self.modality_postprocessors[modality_key](modality_value)

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)
                    
                outputs[modality_key] = modality_value

        return outputs


def imagebind_huge(pretrained=False, store_path=r'.checkpoints'):
    model = ImageBindModel(EncoderArgs())

    if pretrained:
        if not os.path.exists("{}/imagebind_huge.pth".format(store_path)):
            print("Downloading imagebind weights to {}/imagebind_huge.pth ...".format(store_path))
            os.makedirs(store_path, exist_ok=True)
            torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth", "{}/imagebind_huge.pth".format(store_path), progress=True)
        model.load_state_dict(torch.load("{}/imagebind_huge.pth".format(store_path)))
    return model, EncoderArgs.out_embed_dim
