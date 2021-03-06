# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# from megengine.data.dataset import Objects365, PascalVOC
#
# data_mapper = dict(
#     coco=COCO,
#     objects365=Objects365,
#     voc=PascalVOC,
# )

from .NewCOCO import COCO
data_mapper = dict(
    coco=COCO
)