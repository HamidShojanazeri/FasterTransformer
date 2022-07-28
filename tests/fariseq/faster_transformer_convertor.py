from typing import List, Optional, Dict

import numpy as np
import torch
# from accelerators.pytorch.lib.cuda_lowering.nv_faster_transformer import (
#     NVTransformerStack,
# )
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import MultiheadAttention
from fairseq.modules.transformer_layer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from torch import nn, Tensor


def fairseq_to_fast_transformer_decoder_weights(layer):
    self_layernorm_gamma = layer.self_attn_layer_norm.weight
    self_layernorm_beta = layer.self_attn_layer_norm.bias
    self_kernel_q = layer.self_attn.q_proj.weight
    self_kernel_k = layer.self_attn.k_proj.weight
    self_kernel_v = layer.self_attn.v_proj.weight
    self_bias_q = layer.self_attn.q_proj.bias
    self_bias_k = layer.self_attn.k_proj.bias
    self_bias_v = layer.self_attn.v_proj.bias
    self_output_kernel = layer.self_attn.out_proj.weight
    self_output_bias = layer.self_attn.out_proj.bias
    cross_layernorm_gamma = layer.encoder_attn_layer_norm.weight
    cross_layernorm_beta = layer.encoder_attn_layer_norm.bias
    cross_kernel_q = layer.encoder_attn.q_proj.weight
    cross_kernel_k = layer.encoder_attn.k_proj.weight
    cross_kernel_v = layer.encoder_attn.v_proj.weight
    cross_bias_q = layer.encoder_attn.q_proj.bias
    cross_bias_k = layer.encoder_attn.k_proj.bias
    cross_bias_v = layer.encoder_attn.v_proj.bias
    cross_output_kernel = layer.encoder_attn.out_proj.weight
    cross_output_bias = layer.encoder_attn.out_proj.bias
    ffn_layernorm_gamma = layer.final_layer_norm.weight
    ffn_layernorm_beta = layer.final_layer_norm.bias
    inter_kernel = layer.fc1.weight
    inter_bias = layer.fc1.bias
    outer_kernel = layer.fc2.weight
    outer_bias = layer.fc2.bias

    ft_weights = [
        self_layernorm_gamma,
        self_layernorm_beta,
        self_kernel_q.transpose(-1, -2).contiguous(),
        # self_kernel_k.transpose(-1, -2).contiguous(),
        # self_kernel_v.transpose(-1, -2).contiguous(),
        self_bias_q,
        # self_bias_k,
        # self_bias_v,
        self_output_kernel.transpose(-1, -2).contiguous(),
        self_output_bias,
        cross_layernorm_gamma,
        cross_layernorm_beta,
        cross_kernel_q.transpose(-1, -2).contiguous(),
        cross_kernel_k.transpose(-1, -2).contiguous(),
        cross_kernel_v.transpose(-1, -2).contiguous(),
        cross_bias_q,
        cross_bias_k,
        cross_bias_v,
        cross_output_kernel.transpose(-1, -2).contiguous(),
        cross_output_bias,
        ffn_layernorm_gamma,
        ffn_layernorm_beta,
        inter_kernel.transpose(-1, -2).contiguous(),
        inter_bias,
        outer_kernel.transpose(-1, -2).contiguous(),
        outer_bias,
    ]
    return [w.contiguous().cuda() for w in ft_weights]


def fairseq_layer_to_faster_transformer_decoder_layer(layer):
    embed_dim = layer.embed_dim
    num_heads = layer.self_attn.num_heads
    head_dim = embed_dim // num_heads
    scaling = layer.self_attn.scaling
    np.testing.assert_allclose(scaling, 1.0 / np.sqrt(head_dim))

    num_heads = layer.encoder_attn.num_heads
    head_dim = embed_dim // num_heads
    scaling = layer.encoder_attn.scaling
    np.testing.assert_allclose(scaling, 1.0 / np.sqrt(head_dim))

    assert isinstance(layer, TransformerDecoderLayer)
    self_attn = layer.self_attn
    assert isinstance(self_attn, MultiheadAttention)
    encoder_attn = layer.encoder_attn
    assert isinstance(encoder_attn, MultiheadAttention)
    assert not layer.quant_noise
    assert layer.activation_fn.__name__ == "relu"
    assert layer.fc1.in_features * 8 == layer.fc1.out_features
    assert layer.normalize_before
    print(" head_dim : ************", head_dim, head_dim)
    layer_num = 2i
    return torch.classes.FasterTransformer.Decoder(
        *fairseq_to_fast_transformer_decoder_weights(layer),
        num_heads,
        head_dim,
        num_heads * head_dim * 4,
        layer_num,
        embed_dim    
    )

class FairSeqNVFasterTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, old_decoder, beam_width):
        super().__init__(old_decoder.dictionary)
        self.layer_num = len(old_decoder.layers)
        self.beam_width = beam_width
        self.embed_dim = old_decoder.embed_dim
        self.embed_positions = old_decoder.embed_positions
        self.embed_tokens = old_decoder.embed_tokens
        self.embed_scale = old_decoder.embed_scale
        self.layernorm_embedding = old_decoder.layernorm_embedding
        self.layer_norm = old_decoder.layer_norm
        self.dropout_module = old_decoder.dropout_module
        self.output_projection = old_decoder.output_projection
        self.max_target_positions = old_decoder.max_target_positions
        self.layers = [
            fairseq_layer_to_faster_transformer_decoder_layer(layer)
            for layer in old_decoder.layers
        ]

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        assert alignment_layer is None
        bs, step = prev_output_tokens.size()

        assert encoder_out is not None
        enc = encoder_out["encoder_out"][0].transpose(0, 1).contiguous()
        src_lengths = encoder_out["src_lengths"][0].int().contiguous()

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens, incremental_state=incremental_state
        )
        prev_output_tokens = prev_output_tokens[:, -1:]
        positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = self.layernorm_embedding(x)
        x = self.dropout_module(x)

        # dummy attention score [bs x tgt_len x src_len]
        attn: Optional[Tensor] = enc.new_empty([bs, 1, enc.size()[1]])

        # decoder layers
        inner_states: List[Optional[Tensor]] = [x]
        caches: Dict[str, Optional[Tensor]] = {}
        mem_caches: Dict[str, Optional[Tensor]] = {}
        if step == 1:
            for i in range(self.layer_num):
                caches[f"cache_{i}_0"] = enc.new_zeros([1, bs, self.embed_dim])
                caches[f"cache_{i}_1"] = enc.new_zeros([1, bs, self.embed_dim])
        else:
            temp = self.get_incremental_state(incremental_state, "FT_cache")
            assert temp is not None
            caches = temp

            for i in range(self.layer_num):
                cache_0 = caches[f"cache_{i}_0"]
                cache_1 = caches[f"cache_{i}_1"]
                assert cache_0 is not None
                assert cache_1 is not None

                caches[f"cache_{i}_0"] = torch.cat(
                    [cache_0, enc.new_zeros([1, bs, self.embed_dim])], 0
                )
                caches[f"cache_{i}_1"] = torch.cat(
                    [cache_1, enc.new_zeros([1, bs, self.embed_dim])], 0
                )
        self.set_incremental_state(incremental_state, "FT_cache", caches)

        if step == 1:
            for i in range(self.layer_num):
                mem_caches[f"mem_cache_{i}"] = enc.new_empty(
                    (2,) + enc.size()[:-1] + (self.embed_dim,)
                )
            self.set_incremental_state(incremental_state, "FT_mem_cache", mem_caches)
        else:
            temp = self.get_incremental_state(incremental_state, "FT_mem_cache")
            assert temp is not None
            mem_caches = temp

        for i, layer in enumerate(self.layers):
            cache_0 = caches[f"cache_{i}_0"]
            cache_1 = caches[f"cache_{i}_1"]
            mem_cache = mem_caches[f"mem_cache_{i}"]
            assert cache_0 is not None
            assert cache_1 is not None
            assert mem_cache is not None

            # (B1 * B2) x 1 x Hidden
            x = layer.forward(
                x,
                enc,
                src_lengths,
                [cache_0, cache_1],
                mem_cache,
                step - 1,
            )
            inner_states.append(x.transpose(0, 1))

        x = self.layer_norm(x)
        x = self.output_projection(x)
        return x, {"attn": [attn], "inner_states": inner_states}

    def max_positions(self):
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def reorder_incremental_state_scripting(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        caches = self.get_incremental_state(incremental_state, "FT_cache")
        mem_caches = self.get_incremental_state(incremental_state, "FT_mem_cache")
        assert caches is not None
        assert mem_caches is not None

        for i in range(self.layer_num):
            cache_0 = caches[f"cache_{i}_0"]
            cache_1 = caches[f"cache_{i}_1"]
            mem_cache = mem_caches[f"mem_cache_{i}"]
            assert cache_0 is not None
            assert cache_1 is not None
            assert mem_cache is not None

            caches[f"cache_{i}_0"] = cache_0.index_select(1, new_order)
            caches[f"cache_{i}_1"] = cache_1.index_select(1, new_order)
            mem_caches[f"mem_cache_{i}"] = mem_cache.index_select(1, new_order)

        self.set_incremental_state(incremental_state, "FT_cache", caches)
        self.set_incremental_state(incremental_state, "FT_mem_cache", mem_caches)

        for module in self.modules():
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result
