"""
Custom data collator that masks everything except assistant responses.

For GigaChat3-style chat format:
  role<|role_sep|>\ncontent<|message_sep|>\n\n

Only tokens BETWEEN "assistant<|role_sep|>\n" and the next "<|message_sep|>"
are trained on. Everything else (system prompts, user messages, template
boilerplate, DEVSYSTEM) is masked with label=-100.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


def _find_subsequence(seq: list[int], subseq: list[int]) -> list[int]:
    """Find all start positions of subseq in seq."""
    positions = []
    sub_len = len(subseq)
    for i in range(len(seq) - sub_len + 1):
        if seq[i : i + sub_len] == subseq:
            positions.append(i)
    return positions


def build_assistant_mask(
    input_ids: list[int],
    response_marker_ids: list[int],
    message_end_ids: list[int],
) -> list[int]:
    """
    Build labels where only assistant response tokens have actual IDs,
    everything else is IGNORE_INDEX (-100).
    """
    labels = [IGNORE_INDEX] * len(input_ids)

    resp_positions = _find_subsequence(input_ids, response_marker_ids)
    end_positions = _find_subsequence(input_ids, message_end_ids)

    for resp_start in resp_positions:
        content_start = resp_start + len(response_marker_ids)
        content_end = len(input_ids)
        for end_pos in end_positions:
            if end_pos >= content_start:
                content_end = end_pos
                break

        for j in range(content_start, content_end):
            labels[j] = input_ids[j]

    return labels


@dataclass
class AssistantOnlyCollator:
    """
    Pads batch, then masks labels so only assistant responses are trained on.
    Works correctly for multi-turn conversations.
    """

    tokenizer: PreTrainedTokenizerBase
    response_marker_ids: list[int] = field(default_factory=list)
    message_end_ids: list[int] = field(default_factory=list)
    max_seq_length: int | None = None
    mlm: bool = False
    _call_count: int = field(default=0, repr=False, init=False)

    def __call__(self, features: list[dict]) -> dict:
        self._call_count += 1
        labels_list = []
        for feat in features:
            ids = feat["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            labels_list.append(
                build_assistant_mask(
                    ids, self.response_marker_ids, self.message_end_ids,
                )
            )
            feat.pop("labels", None)

        if self._call_count <= 3:
            for i, (lab, feat) in enumerate(zip(labels_list, features)):
                ids = feat["input_ids"]
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                trained = sum(1 for x in lab if x != IGNORE_INDEX)
                print(f"  [Collator batch#{self._call_count} sample#{i}] "
                      f"len={len(ids)}, trained={trained}/{len(lab)}, "
                      f"first_10_ids={ids[:10]}")
                if trained == 0:
                    resp_pos = _find_subsequence(ids, self.response_marker_ids)
                    end_pos = _find_subsequence(ids, self.message_end_ids)
                    print(f"    WARNING: 0 trained tokens! "
                          f"resp_marker_positions={resp_pos}, "
                          f"end_marker_positions={end_pos}")

        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels_list), max_len), IGNORE_INDEX, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            length = min(len(lab), max_len)
            padded_labels[i, :length] = torch.tensor(lab[:length], dtype=torch.long)
        batch["labels"] = padded_labels

        return batch


def get_marker_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[list[int], list[int]]:
    """
    Get token IDs for the response marker and message end marker.
    Returns (response_marker_ids, message_end_ids).
    """
    response_marker = "assistant<|role_sep|>\n"
    message_end = "<|message_sep|>"

    resp_ids = tokenizer.encode(response_marker, add_special_tokens=False)
    end_ids = tokenizer.encode(message_end, add_special_tokens=False)

    return resp_ids, end_ids


def print_masking_diagnostic(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    response_marker_ids: list[int],
    message_end_ids: list[int],
) -> None:
    """Print a visual diagnostic of what tokens are masked vs trained."""
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    labels = build_assistant_mask(input_ids, response_marker_ids, message_end_ids)

    total = len(labels)
    trained = sum(1 for l in labels if l != IGNORE_INDEX)
    masked = total - trained

    print(f"  Masking diagnostic: {total} tokens total, "
          f"{trained} trained ({100 * trained / total:.1f}%), "
          f"{masked} masked ({100 * masked / total:.1f}%)")

    trained_sections = []
    in_section = False
    start = 0
    for i, l in enumerate(labels):
        if l != IGNORE_INDEX and not in_section:
            start = i
            in_section = True
        elif l == IGNORE_INDEX and in_section:
            snippet = tokenizer.decode(input_ids[start:i])
            if len(snippet) > 80:
                snippet = snippet[:40] + " ... " + snippet[-35:]
            trained_sections.append((start, i, snippet))
            in_section = False
    if in_section:
        snippet = tokenizer.decode(input_ids[start:])
        if len(snippet) > 80:
            snippet = snippet[:40] + " ... " + snippet[-35:]
        trained_sections.append((start, len(labels), snippet))

    print(f"  Trained sections ({len(trained_sections)}):")
    for idx, (s, e, snip) in enumerate(trained_sections):
        print(f"    [{idx+1}] tokens {s}-{e}: {snip!r}")
