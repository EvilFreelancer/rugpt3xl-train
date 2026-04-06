"""
Test optimized attention for ruGPT3XL-8k.

Patches the model's attention to use Flash Attention (is_causal=True)
for dense layers and Memory-Efficient backend for sparse layers.
"""

import gc
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def gpu_mem_mb(device=0):
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    free = total - reserved
    return alloc, reserved, free, total


def report(label, device=0):
    alloc, reserved, free, total = gpu_mem_mb(device)
    print(f"[{label}]  alloc={alloc:.0f}MB  reserved={reserved:.0f}MB  free={free:.0f}MB")
    return alloc


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def patch_attention_forward(model):
    """Patch attention to use Flash/MemEfficient backends."""
    import types

    sparse_layers = model.base_model.model.model.model._sparse_layers

    def patched_forward(self, hidden_states, attention_mask=None,
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
                    query, key, value,
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=dropout_p,
                is_causal=True,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None, past_key_value

    layers = model.base_model.model.model.model.layers
    for layer in layers:
        attn = layer.self_attn
        attn.forward = types.MethodType(patched_forward, attn)

    print(f"Patched {len(layers)} attention layers "
          f"(sparse: {sorted(sparse_layers)}, dense: {sorted(set(range(len(layers))) - sparse_layers)})")


def main():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print()

    flush()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/workspace/models/ruGPT3XL-8k",
        max_seq_length=8192,
        load_in_4bit=True,
        dtype=None,
        trust_remote_code=True,
    )
    report("Model loaded (4-bit)")

    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    report("LoRA applied")

    patch_attention_forward(model)
    print()

    model.gradient_checkpointing_enable()
    model.train()
    report("Training mode + grad checkpointing")

    seq_lengths = [4096, 6144, 8192]

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len}")
        print(f"{'='*60}")

        flush()

        input_ids = torch.randint(0, 50264, (1, seq_len), device=device)
        labels = input_ids.clone()

        try:
            torch.cuda.reset_peak_memory_stats()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            peak_fwd = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  forward peak: {peak_fwd:.0f} MB, loss: {loss.item():.4f}")

            torch.cuda.reset_peak_memory_stats()
            loss.backward()
            peak_bwd = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  backward peak: {peak_bwd:.0f} MB")

            model.zero_grad(set_to_none=True)
            del outputs, loss, input_ids, labels
            flush()

            print(f"  >>> OK! peak_fwd={peak_fwd:.0f}MB, peak_bwd={peak_bwd:.0f}MB")

        except torch.cuda.OutOfMemoryError:
            print(f"  >>> OOM at seq_len={seq_len}!")
            del input_ids, labels
            flush()

        except Exception as e:
            print(f"  >>> Error: {e}")
            import traceback
            traceback.print_exc()
            del input_ids, labels
            flush()

    print("\nDone!")


if __name__ == "__main__":
    main()
