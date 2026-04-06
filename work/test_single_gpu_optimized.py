"""Test optimized attention on single GPU for 8k context."""

import gc
import os
import types

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def gpu_mem():
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    peak = torch.cuda.max_memory_allocated(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return alloc, peak, total


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def patch_model(model):
    """Patch both model-level forward (avoid building masks) and attention."""
    base = model.base_model.model.model
    sparse_layers = base._sparse_layers

    # Patch attention forward
    def patched_attn_forward(self, hidden_states, attention_mask=None,
                             position_ids=None, past_key_value=None,
                             output_attentions=False, use_cache=False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if past_key_value is not None:
            key, value = past_key_value.update(key, value, self.layer_idx)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        is_sparse = self.layer_idx in sparse_layers
        if is_sparse and attention_mask is not None:
            sdpa_mask = attention_mask.to(dtype=query.dtype)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_output = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=sdpa_mask,
                    dropout_p=dropout_p, is_causal=False,
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=True,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, None, past_key_value

    for layer in base.layers:
        layer.self_attn.forward = types.MethodType(patched_attn_forward, layer.self_attn)

    # Patch model forward to skip building causal mask for dense layers
    original_forward = base.forward.__func__ if hasattr(base.forward, '__func__') else base.forward

    def patched_model_forward(self, input_ids=None, attention_mask=None,
                              position_ids=None, past_key_values=None,
                              inputs_embeds=None, use_cache=None,
                              output_attentions=None, output_hidden_states=None,
                              return_dict=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.cache_utils import DynamicCache

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_embeds = self.embed_positions(position_ids)
        hidden_states = self.embed_dropout(inputs_embeds + position_embeds)

        # Only build sparse mask (skip dense causal mask - use is_causal=True)
        sparse_mask = None
        if self._sparse_layers:
            sparse_layout = self._get_sparse_layout(hidden_states.device)
            sparse_mask = self._build_sparse_causal_mask(
                seq_length, past_key_values_length,
                hidden_states.dtype, hidden_states.device,
                sparse_layout, self.config.sparse_block_size,
                attention_mask,
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Sparse layers get the sparse mask, dense layers get None (use is_causal)
            layer_mask = sparse_mask if layer_idx in self._sparse_layers else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states, layer_mask, position_ids,
                    past_key_values, output_attentions, use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=layer_mask,
                    position_ids=position_ids, past_key_value=past_key_values,
                    output_attentions=output_attentions, use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=next_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns,
        )

    base.forward = types.MethodType(patched_model_forward, base)
    print("  Patched model forward + 24 attention layers")


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

print("\n[1] Loading model on single GPU ...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/models/ruGPT3XL-8k",
    max_seq_length=8192, load_in_4bit=True, dtype=None,
    trust_remote_code=True,
)
alloc, peak, total = gpu_mem()
print(f"  Model loaded: {alloc:.0f}MB allocated")

print("\n[2] LoRA + patch ...")
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
)
patch_model(model)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

for seq_len in [4096, 6144, 8192]:
    print(f"\n[Test] seq_len={seq_len} ...")
    flush()
    device = torch.device("cuda:0")
    input_ids = torch.randint(0, 50264, (1, seq_len), device=device)
    labels = input_ids.clone()

    try:
        torch.cuda.reset_peak_memory_stats()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        fwd_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Forward: peak={fwd_peak:.0f}MB, loss={loss.item():.4f}")

        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        bwd_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Backward: peak={bwd_peak:.0f}MB")

        model.zero_grad(set_to_none=True)
        del outputs, loss, input_ids, labels
        flush()
        print(f"  >>> PASSED (fwd={fwd_peak:.0f}MB, bwd={bwd_peak:.0f}MB)")

    except torch.cuda.OutOfMemoryError:
        print(f"  >>> OOM!")
        del input_ids, labels
        flush()

    except Exception as e:
        print(f"  >>> Error: {e}")
        import traceback
        traceback.print_exc()
        del input_ids, labels
        flush()

print("\nDone!")
