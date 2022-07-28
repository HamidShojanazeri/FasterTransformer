from abc import ABC
import json
import logging
import os
import ast
import torch
from ts.torch_handler.base_handler import BaseHandler
from pathlib import Path
from inference.eval_prompts import Generator
from fairseq.utils import set_torch_seed
from dataclasses import dataclass
from typing import Tuple
import time
from pkg_resources import packaging
from faster_transformer_convertor import FairSeqNVFasterTransformerDecoder, fairseq_layer_to_faster_transformer_decoder_layer, fairseq_to_fast_transformer_decoder_weights
from typing import Any, Dict, Sequence
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.models import transformer_lm
from fairseq.models.transformer import TransformerModel
import fairseq
import unittest
from fairseq.data import Dictionary
import argparse
import argparse
import tempfile
import unittest
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn
from fairseq.data import Dictionary
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
)
def load_model():

    serialized_file = "checkpoint.pt"

    device = "cuda:0"
    print("device", device )
    tokenizer_file = str("galileo-6.7b/tokenizer.json")
    generator = Generator("./galileo-6.7b", serialized_file, tokenizer_file, use_cuda=False)
    if device != "cpu":

        generator.model.half().to(device)

    generator.model.models[0].decoder.max_target_positions = 4096
    generator.model.eval()
    generator.tokenizer.enable_padding(pad_id=generator.tokenizer.token_to_id("<pad>"))
    return generator


SRC_DICT_SIZE = 123
TGT_DICT_SIZE = 120
ENCODER_EMBED_DIM = 256
DECODER_EMBED_DIM = 128
RESERVED_TOKENS_COUNT = 4
PAD = Dictionary().pad()
EOS = Dictionary().eos()

def _generate_dummy_dictionary(vocab_size: int):
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as dict_file:
        for i in range(vocab_size - RESERVED_TOKENS_COUNT):
            dict_file.write(f"{i} 1\n")
        dict_file.flush()
        return Dictionary.load(dict_file.name)

def build_dummy_encoder(encoder_normalize_before=True):
    config = argparse.Namespace(
        _name="transformer",
        activation_dropout=0.0,
        activation_fn="relu",
        adaptive_input=False,
        arch="transformer",
        attention_dropout=0.1,
        cross_self_attention=False,
        dropout=0.2,
        encoder_attention_heads=16,
        encoder_embed_dim=ENCODER_EMBED_DIM,
        encoder_ffn_embed_dim=256,
        encoder_layerdrop=0,
        encoder_layers=2,
        encoder_layers_to_keep=None,
        encoder_learned_pos=False,
        encoder_normalize_before=encoder_normalize_before,
        eos=2,
        export=True,
        layernorm_embedding=True,
        max_source_positions=300,
        max_target_positions=300,
        no_cross_attention=False,
        no_scale_embedding=False,
        no_token_positional_embeddings=False,
        quant_noise_pq=0,
        quant_noise_pq_block_size=8,
        quant_noise_scalar=0,
        seed=0,
        share_all_embeddings=False,
        share_decoder_input_output_embed=False,
        unk=3,
    )
    src_dictionary = _generate_dummy_dictionary(SRC_DICT_SIZE)
    encoder_embedding = TransformerModel.build_embedding(
        None, src_dictionary, config.encoder_embed_dim
    )
    return TransformerEncoder(config, src_dictionary, encoder_embedding)

def build_dummy_decoder():
    config = argparse.Namespace(
        _name="transformer",
        activation_dropout=0.0,
        activation_fn="relu",
        adaptive_input=False,
        adaptive_softmax_cutoff=None,
        arch="transformer",
        attention_dropout=0.1,
        cross_self_attention=False,
        dropout=0.2,
        encoder_attention_heads=16,
        encoder_embed_dim=ENCODER_EMBED_DIM,
        encoder_ffn_embed_dim=ENCODER_EMBED_DIM,
        encoder_layerdrop=0,
        encoder_layers=2,
        encoder_layers_to_keep=None,
        encoder_learned_pos=False,
        encoder_normalize_before=True,
        decoder_attention_heads=16,
        decoder_embed_dim=DECODER_EMBED_DIM,
        decoder_embed_path=None,
        decoder_ffn_embed_dim=8 * DECODER_EMBED_DIM,
        decoder_input_dim=DECODER_EMBED_DIM,
        decoder_layerdrop=0,
        decoder_layers=2,
        decoder_layers_to_keep=None,
        decoder_learned_pos=False,
        decoder_normalize_before=True,
        decoder_output_dim=DECODER_EMBED_DIM,
        eos=2,
        export=True,
        layernorm_embedding=True,
        max_source_positions=300,
        max_target_positions=300,
        no_cross_attention=False,
        no_scale_embedding=False,
        no_token_positional_embeddings=False,
        quant_noise_pq=0,
        quant_noise_pq_block_size=8,
        quant_noise_scalar=0,
        seed=0,
        share_all_embeddings=False,
        share_decoder_input_output_embed=False,
        tie_adaptive_weights=False,
        unk=3,
    )
    tgt_dictionary = _generate_dummy_dictionary(TGT_DICT_SIZE)
    decoder_embedding = TransformerModel.build_embedding(
        None, tgt_dictionary, config.decoder_embed_dim
    )
    return TransformerDecoder(config, tgt_dictionary, decoder_embedding)
if __name__ == "__main__":
    # generator = load_model()
    model = build_dummy_decoder()
    print(model)
    layers = [

        fairseq_layer_to_faster_transformer_decoder_layer(layer)
        for layer in model.layers
    ]

def testLoweringTransformerDecoderToFairSeqNVFasterTransformerDecoder():
        encoder = build_dummy_encoder(encoder_normalize_before=True)
        decoder = build_dummy_decoder()
        encoder = encoder.cuda().half().eval()
        decoder = decoder.cuda().half().eval()
        ft_decoder = FairSeqNVFasterTransformerDecoder(decoder, 1).cuda().half().eval()

        # Somehow incremental_state doesn't work in unit test after script
        # ft_decoder = torch.jit.script(ft_decoder)
        # # Serialize and deserialize
        with tempfile.NamedTemporaryFile(mode="wb", delete=True) as save_to:
            torch.save(ft_decoder, save_to.name)
            ft_decoder = torch.load(save_to.name)

        INPUT_SEQ_LEN = 10
        BATCH_SIZE = 1
        src_tokens = (
            torch.randint(
                low= RESERVED_TOKENS_COUNT + 1,
                high= SRC_DICT_SIZE - 1,
                size=(BATCH_SIZE, INPUT_SEQ_LEN),
            )
            .cuda()
            .long()
        )
        src_tokens[:, -1] = EOS
        src_lengths = (
            torch.tensor([INPUT_SEQ_LEN] * BATCH_SIZE)
            .cuda()
            .long()
            .reshape(BATCH_SIZE, 1)
            .contiguous()
        )
        tokens = (
            torch.zeros(1, 10).to(src_tokens).long().fill_(10)
        )  # +2 for eos and pad
        encoder_out = encoder(src_tokens)
        encoder_out["src_lengths"] = [src_lengths]
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[torch.Tensor]]], {}
        )
        ft_incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[torch.Tensor]]], {}
        )

        for step in range(8):
            if step > 0:
                tokens[:, step - 1] = step + RESERVED_TOKENS_COUNT + 5
            ref_x, ref_extra = decoder(
                tokens[:, : step + 1], encoder_out, incremental_state
            )
            y_x, y_extra = ft_decoder(
                tokens[:, : step + 1], encoder_out, ft_incremental_state
            )
            for rref, yy in zip(ref_extra["inner_states"], y_extra["inner_states"]):
                torch.testing.assert_allclose(rref, yy, atol=1e-1, rtol=5e-2)
