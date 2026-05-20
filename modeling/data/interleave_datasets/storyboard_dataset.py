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


class StoryboardDataset(InterleavedBaseIterableDataset, JSONLStandardIterableDataset):

    def parse_row(self, row, ablation=""):
        
        img_root = row["img_root"]
        img_list = row["img_list"]
        narratives = row["narratives"]
        schemas = row["schemas"]

        data = self._init_data()
        for idx in range(0, len(narratives)):
            data = self._add_text(
                data,
                narratives[idx],
                need_loss=False,
                schema_ce_reweight=False,
                prompt=True)

            data = self._add_text(
                data,
                schemas[idx],
                need_loss=True,
                schema_ce_reweight=True,
                vwm=True)
            
            if idx < len(narratives)-1:
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(os.path.join(img_root, img_list[idx]))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=True,
                    vwm=True)
            else:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(os.path.join(img_root, img_list[idx]))),
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                    vwm=True)

        return data
