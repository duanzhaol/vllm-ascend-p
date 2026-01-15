#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""
Sharded state loader with multi-thread support for Ascend NPU.

This module extends vLLM's ShardedStateLoader to support multi-threaded
weight loading, which can significantly speed up model loading when
using pre-sharded checkpoints.

Usage:
    vllm serve /path/to/sharded_model \
        --load-format sharded_state \
        --tensor-parallel-size 4 \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'
"""

from collections.abc import Generator

import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.sharded_state_loader import (
    ShardedStateLoader,
)
from vllm.model_executor.model_loader.weight_utils import (
    multi_thread_safetensors_weights_iterator,
    runai_safetensors_weights_iterator,
)


DEFAULT_NUM_THREADS = 8


@register_model_loader("sharded_state")
class AscendShardedStateLoader(ShardedStateLoader):
    """
    Sharded state loader with multi-thread support for Ascend NPU.

    This loader extends the base ShardedStateLoader to support multi-threaded
    weight loading via the `enable_multithread_load` and `num_threads` options
    in `model_loader_extra_config`.
    """

    def __init__(self, load_config: LoadConfig):
        # Don't call super().__init__() directly since it will raise error
        # for extra config keys. Instead, handle the config ourselves.
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        BaseModelLoader.__init__(self, load_config)

        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )

        # Extract sharded_state specific config
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)

        # Extract multi-thread config
        self.enable_multithread_load = extra_config.pop(
            "enable_multithread_load", False
        )
        self.num_threads = extra_config.pop("num_threads", DEFAULT_NUM_THREADS)

        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{extra_config.keys()}"
            )

    def iterate_over_files(
        self, paths
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Iterate over the weights in the sharded checkpoint files."""
        if self.load_config.load_format == "runai_streamer_sharded":
            yield from runai_safetensors_weights_iterator(paths, True)
        elif self.enable_multithread_load:
            # Use multi-threaded loading
            yield from multi_thread_safetensors_weights_iterator(
                paths,
                use_tqdm_on_load=self.load_config.use_tqdm_on_load,
                max_workers=self.num_threads,
            )
        else:
            # Default single-threaded loading
            from safetensors.torch import safe_open

            for path in paths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        yield key, tensor
