import hydra_zen
import torch
from transformers import AutoConfig
from transformers.models.auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer


def load_model(
    pretrained_model_name_or_path: str,
    from_config: bool = False,
    seed: int | None = None,
    **kwargs,
):
    if from_config:
        if seed is not None:
            torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config_kwargs = {k: v for k, v in kwargs.items() if k in ['torch_dtype', 'attn_implementation']}
        return AutoModelForCausalLM.from_config(config, **config_kwargs)
    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)


@hydra_zen.hydrated_dataclass(
    target=AutoTokenizer.from_pretrained,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class TokenizerConfig:
    pretrained_model_name_or_path: str
    force_download: bool = False
    local_files_only: bool = False
    token: str | bool | None = None
    revision: str = "main"
    use_fast: bool = True
    subfolder: str = ""
    tokenizer_type: str | None = None
    trust_remote_code: bool = False

@hydra_zen.hydrated_dataclass(
    target=load_model,
    frozen=True,
    unsafe_hash=True,
)
class NetworkConfig:
    pretrained_model_name_or_path: str
    from_config: bool = False
    seed: int | None = None
    trust_remote_code: bool = False
    torch_dtype: torch.dtype | None = None
    attn_implementation: str | None = None
    revision: str = "main"
    low_cpu_mem_usage: bool | None = None
    quantization_config: object | None = None
    device_map: object | None = None
