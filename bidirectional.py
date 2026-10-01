"""Turn a Qwen3 decoder into a bidirectional encoder (full attention, no causal mask).

Two things make Qwen3 causal in transformers (checked on 5.18 by perturbing a future token and
spying on the SDPA kernel call):
  1. Qwen3Model builds its mask with `create_causal_mask`. Used by eager always, and by SDPA when
     there is padding (a mask is materialized and `is_causal` is ignored).
  2. Each attention layer has `is_causal=True`. Used by SDPA when there is no padding: the mask is
     skipped (None) and the kernel is called with the module's `is_causal`. Ignored by eager.
Both must be switched off; either one alone leaves some path causal.

The mask builder is a module-level name, so the patch applies to every Qwen3 model in the process.
Call `make_bidirectional` again after reloading a saved checkpoint.
"""

from transformers.masking_utils import create_bidirectional_mask
from transformers.models.qwen3 import modeling_qwen3


def make_bidirectional(model):
    if model.config.layer_types and any(t != "full_attention" for t in model.config.layer_types):
        raise ValueError("Sliding-window layers are not supported; expected full attention only.")
    modeling_qwen3.create_causal_mask = create_bidirectional_mask
    for module in model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
    model.config.is_bidirectional = True  # informational, saved in config.json
    return model
