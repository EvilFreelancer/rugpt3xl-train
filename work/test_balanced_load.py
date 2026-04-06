"""Quick test: load model with device_map=balanced, run 8k forward+backward."""

import gc
import os
import types

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def gpu_report():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2
        peak = torch.cuda.max_memory_allocated(i) / 1024**2
        print(f"  GPU{i}: alloc={alloc:.0f}MB  peak={peak:.0f}MB / {total:.0f}MB")


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def patch_optimized_attention(base_model):
    sparse_layers = base_model._sparse_layers

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

    for layer in base_model.layers:
        layer.self_attn.forward = types.MethodType(patched_attn_forward, layer.self_attn)
    print(f"  Patched {len(base_model.layers)} attention layers")


print(f"GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")

print("\n[1] Loading model (device_map=balanced) ...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/models/ruGPT3XL-8k",
    max_seq_length=8192,
    load_in_4bit=True,
    dtype=None,
    trust_remote_code=True,
    device_map="balanced",
)
print("  Model loaded:")
gpu_report()

print("\n[2] Applying LoRA ...")
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
)
gpu_report()

print("\n[3] Patching attention ...")
base_model = model.base_model.model.model
patch_optimized_attention(base_model)

print("\n[4] Enabling gradient checkpointing + train mode ...")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

print("\n[5] Layer device placement:")
for i, layer in enumerate(base_model.layers):
    dev = next(layer.parameters()).device
    kind = "sparse" if i in base_model._sparse_layers else "dense"
    print(f"  layer {i:2d} ({kind:6s}) -> {dev}")

# Test forward+backward at 8192
for seq_len in [4096, 8192]:
    print(f"\n[6] Testing seq_len={seq_len} ...")
    flush()
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 50264, (1, seq_len), device=device)
    labels = input_ids.clone()

    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"  Forward OK, loss={loss.item():.4f}")
        gpu_report()

        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

        loss.backward()
        print(f"  Backward OK!")
        gpu_report()

        model.zero_grad(set_to_none=True)
        del outputs, loss, input_ids, labels
        flush()
        print(f"  >>> seq_len={seq_len} PASSED!")

    except torch.cuda.OutOfMemoryError as e:
        print(f"  >>> OOM at seq_len={seq_len}: {e}")
        del input_ids, labels
        flush()

    except Exception as e:
        print(f"  >>> Error at seq_len={seq_len}: {e}")
        import traceback
        traceback.print_exc()
        del input_ids, labels
        flush()

print("\nDone!")
