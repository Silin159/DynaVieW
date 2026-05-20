# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, JSONLStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class StateTransitionDataset(InterleavedBaseIterableDataset, JSONLStandardIterableDataset):

    def parse_row(self, row, ablation=""):
        
        img_root = row["img_root"]
        txt_root = row["txt_root"]
        img_list = row["img_list"]
        txt_list = row["txt_list"]
        assert len(img_list) == len(txt_list)+1

        if not (ablation == "no_prev_trans" or ablation == "no_history"):
            vwm = True
        else:
            vwm = False

        data = self._init_data()
        if len(txt_list) == 1:
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(os.path.join(img_root, img_list[0]))),
                need_loss=False,
                need_vae=False,
                need_vit=True,
                vwm=vwm,
            )
        else:
            if not ablation == "no_history":
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(os.path.join(img_root, img_list[0]))),
                    need_loss=False,
                    need_vae=False,
                    need_vit=True,
                    vwm=vwm,
                )

            for idx in range(0, len(txt_list)-1):
                if not (ablation == "no_prev_trans" or ablation == "no_history"):
                    with open(os.path.join(txt_root, txt_list[idx]), "r") as f:
                        st = json.load(f)
                    data = self._add_text(
                        data,
                        st["transition"],
                        need_loss=True,
                        schema_ce_reweight=True,
                        vwm=vwm,
                    )
                
                if idx == len(txt_list)-2:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(os.path.join(img_root, img_list[idx+1]))),
                        need_loss=True,
                        need_vae=False,
                        need_vit=True,
                        vwm=vwm,
                    )
                else:
                    if not ablation == "no_history":
                        data = self._add_image(
                            data, 
                            pil_img2rgb(Image.open(os.path.join(img_root, img_list[idx+1]))),
                            need_loss=True,
                            need_vae=False,
                            need_vit=True,
                            vwm=vwm,
                        )
        
        with open(os.path.join(txt_root, txt_list[-1]), "r") as f:
            st = json.load(f)
        data = self._add_text(
            data,
            st["transition"],
            need_loss=True,
            schema_ce_reweight=True,
            vwm=vwm,
        )
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(os.path.join(img_root, img_list[-1]))),
            need_loss=True,
            need_vae=False,
            need_vit=False,
            vwm=vwm,
        )

        return data


class TransitionDataset(InterleavedBaseIterableDataset, JSONLStandardIterableDataset):

    def parse_row(self, row, ablation=""):
        
        img_root = row["img_root"]
        txt_root = row["txt_root"]
        img_list = row["img_list"]
        txt_list = row["txt_list"]
        assert len(img_list) == len(txt_list)+1
        
        if not (ablation == "no_prev_trans" or ablation == "no_history"):
            vwm = True
        else:
            vwm = False

        data = self._init_data()
        if len(txt_list) == 1:
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(os.path.join(img_root, img_list[0]))),
                need_loss=False,
                need_vae=False,
                need_vit=True,
                vwm=vwm,
            )
        else:
            if not ablation == "no_history":
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(os.path.join(img_root, img_list[0]))),
                    need_loss=False,
                    need_vae=False,
                    need_vit=True,
                    vwm=vwm,
                )

            for idx in range(0, len(txt_list)-1):
                if not (ablation == "no_prev_trans" or ablation == "no_history"):
                    with open(os.path.join(txt_root, txt_list[idx]), "r") as f:
                        st = json.load(f)
                    data = self._add_text(
                        data,
                        st["transition"],
                        need_loss=True,
                        schema_ce_reweight=True,
                        vwm=vwm,
                    )
                
                if idx == len(txt_list)-2:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(os.path.join(img_root, img_list[idx+1]))),
                        need_loss=False,
                        need_vae=False,
                        need_vit=True,
                        vwm=vwm,
                    )
                else:
                    if not ablation == "no_history":
                        data = self._add_image(
                            data, 
                            pil_img2rgb(Image.open(os.path.join(img_root, img_list[idx+1]))),
                            need_loss=False,
                            need_vae=False,
                            need_vit=True,
                            vwm=vwm,
                        )
        
        with open(os.path.join(txt_root, txt_list[-1]), "r") as f:
            st = json.load(f)
        data = self._add_text(
            data,
            st["transition"],
            need_loss=True,
            schema_ce_reweight=True,
            vwm=vwm,
        )
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(os.path.join(img_root, img_list[-1]))),
            need_loss=True,
            need_vae=False,
            need_vit=False,
            vwm=vwm,
        )

        return data
