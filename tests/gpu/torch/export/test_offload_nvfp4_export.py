# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check NVFP4 AWQ export with real CPU offload and CUDA calibration."""

import copy

import pytest
import torch
from accelerate import dispatch_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback


@pytest.mark.parametrize("excluded_consumer", [False, True])
@pytest.mark.parametrize("offload_location", ["cpu", "disk"])
@pytest.mark.timeout(300)
def test_offloaded_nvfp4_awq_export(tmp_path, excluded_consumer, offload_location):
    torch.manual_seed(1234)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForCausalLM"],
    )
    original = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for module in original.modules():
            weight = getattr(module, "weight", None)
            if weight is not None and weight.ndim == 1:
                weight.copy_(torch.randn_like(weight) * 0.25)
    quant_cfg = copy.deepcopy(mtq.NVFP4_AWQ_LITE_CFG)
    if excluded_consumer:
        quant_cfg["quant_cfg"].append({"quantizer_name": "*self_attn.v_proj*", "enable": False})
    tokens = torch.randint(0, 128, (1, 16), device="cuda")

    def calibrate(model):
        with torch.no_grad():
            model(input_ids=tokens, use_cache=False)

    exports = []
    logits = []
    for offload in (False, True):
        model = copy.deepcopy(original).cuda()
        if offload:
            model = dispatch_model(
                model,
                device_map={
                    "model.embed_tokens": 0,
                    "model.layers": offload_location,
                    "model.norm": 0,
                    "lm_head": 0,
                },
                main_device=torch.device("cuda:0"),
                force_hooks=True,
                offload_dir=str(tmp_path / "weights_offload"),
            )
            assert model.model.layers[0].input_layernorm.weight.is_meta
        mtq.quantize(model, quant_cfg, calibrate)
        with torch.no_grad():
            output = model(input_ids=tokens, use_cache=False).logits.detach().cpu()
        assert torch.isfinite(output).all()
        logits.append(output)
        layer = model.model.layers[0]
        expected_norms = {}
        attention_members = (layer.self_attn.q_proj, layer.self_attn.k_proj)
        if not excluded_consumer:
            attention_members += (layer.self_attn.v_proj,)
        for norm, members, guarded in (
            ("input_layernorm", attention_members, excluded_consumer),
            ("post_attention_layernorm", (layer.mlp.gate_proj, layer.mlp.up_proj), False),
        ):
            norm_module = layer.get_submodule(norm)
            with enable_weight_access_and_writeback(norm_module, model, None, writeback=False):
                weight = norm_module.weight.detach().clone()
                scale = torch.stack([m.input_quantizer.pre_quant_scale for m in members]).mean(0)
                expected_norms[norm] = (
                    weight if guarded else ((1 + weight) * scale - 1).to(weight.dtype)
                ).cpu()
        export_dir = tmp_path / ("offloaded" if offload else "resident")
        rows, columns = layer.self_attn.q_proj.weight.shape
        export_hf_checkpoint(model, export_dir=str(export_dir))
        tensors = {
            key: value
            for shard in export_dir.glob("*.safetensors")
            for key, value in load_file(str(shard)).items()
        }
        assert tensors, "Export must contain weights"
        layer_tensors = {
            key.split("layers.0.", 1)[1]: value
            for key, value in tensors.items()
            if "layers.0." in key
        }
        for norm, expected in expected_norms.items():
            torch.testing.assert_close(layer_tensors[f"{norm}.weight"], expected, rtol=0, atol=0)
        assert layer_tensors["self_attn.q_proj.weight"].dtype == torch.uint8
        assert layer_tensors["self_attn.q_proj.weight"].shape == (rows, columns // 2)
        assert layer_tensors["self_attn.q_proj.weight_scale"].shape == (rows, columns // 16)
        exports.append(tensors)
        del model
    torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
    assert exports[0].keys() == exports[1].keys()
    for key in exports[0]:
        torch.testing.assert_close(exports[0][key].float(), exports[1][key].float(), rtol=0, atol=0)
