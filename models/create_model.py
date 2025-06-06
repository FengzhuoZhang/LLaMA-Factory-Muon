import torch
import torch.nn as nn
import math
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from pathlib import Path
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_debugpy



def init_llama_weights(module):
    """
    Initialize weights for Llama model following the original paper's initialization scheme.
    
    This function implements proper weight initialization for:
    - Linear layers (including attention projections)
    - Embedding layers
    - Layer normalization
    """
    
    if isinstance(module, nn.Linear):
        # Initialize linear layers with Xavier uniform (also called Glorot uniform)
        # Standard deviation = sqrt(2 / (fan_in + fan_out))
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        # Initialize embeddings with normal distribution
        # Standard deviation based on hidden size
        std = 1.0 / math.sqrt(module.embedding_dim)
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        
        # Zero out padding token embedding if it exists
        if hasattr(module, 'padding_idx') and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)
    
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        # Initialize layer norm weights to 1 and bias to 0
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def initialize_llama_model(config_path_or_dict=None, **config_kwargs):
    """
    Create and initialize a Llama model with proper weight initialization.
    
    Args:
        config_path_or_dict: Either a path to config file or a config dict
        **config_kwargs: Additional config parameters
    
    Returns:
        Initialized LlamaForCausalLM model
    """
    
    # Create configuration
    if config_path_or_dict is None:
        # Default small config for testing/demonstration
        config = LlamaConfig(
            use_cache=True,
            tie_word_embeddings=False,
            **config_kwargs
        )
    elif isinstance(config_path_or_dict, dict):
        config = LlamaConfig(**config_path_or_dict, **config_kwargs)
    else:
        config = LlamaConfig.from_pretrained(config_path_or_dict, **config_kwargs)
    
    # Create model
    model = LlamaForCausalLM(config)
    
    # Apply weight initialization
    model.apply(init_llama_weights)
    
    # Special initialization for output projection
    # Scale down the output layer weights for better training stability
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        std = 1.0 / math.sqrt(config.hidden_size)
        torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=std)
    
    return model


def get_parameter_count(model):
    """
    Calculate total number of parameters in the model.
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_millions': total_params / 1e6,
        'trainable_parameters_millions': trainable_params / 1e6
    }

def get_tokenizer_info(tokenizer_path_or_name):
    """
    Extract key information from an existing tokenizer.
    
    Args:
        tokenizer_path_or_name: Path to tokenizer or HuggingFace model name
    
    Returns:
        Dictionary with tokenizer information
    """
    try:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path_or_name)
        
        info = {
            'vocab_size': len(tokenizer),
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'bos_token': tokenizer.bos_token,
            'eos_token': tokenizer.eos_token,
            'pad_token': tokenizer.pad_token,
            'unk_token': tokenizer.unk_token,
        }
        
        print(f"Tokenizer Info for {tokenizer_path_or_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return info, tokenizer
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None
    
def create_matched_model_config(tokenizer_info, base_config=None):
    """
    Create a model config that matches the tokenizer.
    
    Args:
        tokenizer_info: Dictionary from get_tokenizer_info()
        base_config: Base configuration dict to modify
    
    Returns:
        LlamaConfig with matching vocabulary size
    """
    
    if base_config is None:
        # Default Llama-2 7B configuration
        base_config = {
            'vocab_size': 32000,
            'hidden_size': 768,
            'intermediate_size': 4*768,
            'num_hidden_layers': 12,
            'num_attention_heads': 6,
            'num_key_value_heads': 6,  # For grouped-query attention
            'hidden_act': "silu",
            'max_position_embeddings': 2048,
            'initializer_range': 0.02,
            'rms_norm_eps': 1e-6,
            'rope_theta': 10000.0,
        }
    
    # Match tokenizer vocabulary size
    config_dict = base_config.copy()
    config_dict.update({
        'vocab_size': tokenizer_info['vocab_size'],
        'bos_token_id': tokenizer_info['bos_token_id'],
        'eos_token_id': tokenizer_info['eos_token_id'],
        'pad_token_id': tokenizer_info['pad_token_id'],
    })
    
    return LlamaConfig(**config_dict)

def initialize_special_tokens(model, tokenizer_info):
    """
    Properly initialize embeddings for special tokens.
    
    Args:
        model: LlamaForCausalLM model
        tokenizer_info: Token information from tokenizer
    """
    
    embed_tokens = model.model.embed_tokens
    lm_head = model.lm_head
    
    with torch.no_grad():
        # Initialize special tokens with small random values
        special_token_ids = [
            tokenizer_info['bos_token_id'],
            tokenizer_info['eos_token_id'], 
            tokenizer_info['pad_token_id'],
            tokenizer_info['unk_token_id']
        ]
        
        for token_id in special_token_ids:
            if token_id is not None and token_id < model.config.vocab_size:
                # Initialize with small random values
                embed_tokens.weight[token_id] = torch.randn_like(embed_tokens.weight[token_id]) * 0.01
                lm_head.weight[token_id] = torch.randn_like(lm_head.weight[token_id]) * 0.01
        
        # Set padding token embedding to zero
        if tokenizer_info['pad_token_id'] is not None:
            embed_tokens.weight[tokenizer_info['pad_token_id']].fill_(0)
    
    print("Special tokens initialized")

def verify_model_tokenizer_compatibility(model, tokenizer):
    """
    Verify that model and tokenizer are compatible.
    
    Args:
        model: LlamaForCausalLM model
        tokenizer: LlamaTokenizer
    """
    
    print("\nVerifying model-tokenizer compatibility...")
    
    # Check vocabulary sizes match
    model_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = len(tokenizer)
    
    assert model_vocab_size == tokenizer_vocab_size, \
        f"Vocab size mismatch: model={model_vocab_size}, tokenizer={tokenizer_vocab_size}"
    
    # Check special token IDs match
    config_bos = getattr(model.config, 'bos_token_id', None)
    config_eos = getattr(model.config, 'eos_token_id', None)
    config_pad = getattr(model.config, 'pad_token_id', None)
    
    assert config_bos == tokenizer.bos_token_id, \
        f"BOS token mismatch: model={config_bos}, tokenizer={tokenizer.bos_token_id}"
    assert config_eos == tokenizer.eos_token_id, \
        f"EOS token mismatch: model={config_eos}, tokenizer={tokenizer.eos_token_id}"
    
    # Test tokenization and model forward pass
    test_text = "Hello, how are you today?"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✓ Vocabulary sizes match: {model_vocab_size}")
    print(f"✓ Special tokens match")
    print(f"✓ Forward pass successful: {outputs.logits.shape}")
    print(f"✓ Test text tokenized to {len(inputs.input_ids[0])} tokens")

def save_matched_model_huggingface_format(model, tokenizer, save_path, model_name="custom-llama", max_shard_size="5GB"):
    """
    Save the matched model and tokenizer in complete Hugging Face format with sharded safetensors.
    
    Args:
        model: LlamaForCausalLM model
        tokenizer: LlamaTokenizer
        save_path: Path to save
        model_name: Name for the model card
        max_shard_size: Maximum size per shard (e.g., "5GB", "2GB")
    """
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model in Hugging Face sharded safetensors format to: {save_path}")
    
    # 1. Save model weights in sharded safetensors format with index
    model.save_pretrained(
        save_path,
        safe_serialization=True,  # Use safetensors instead of pickle
        max_shard_size=max_shard_size  # Shard the model
    )
    print("✓ Model weights saved as sharded safetensors")
    print("✓ Model config saved (config.json)")
    print("✓ Model index saved (model.safetensors.index.json)")
    
    # 2. Save tokenizer files (no tokenizer.model for official format)
    save_tokenizer_official_format(tokenizer, save_path)
    print("✓ Tokenizer files saved (official format)")
    
    # 3. Create generation config
    create_generation_config_hf(save_path, model.config)
    print("✓ Generation config created")
    
    # # 4. Create model card (README.md)
    # create_model_card_hf(save_path, model, tokenizer, model_name)
    # print("✓ Model card (README.md) created")
    
    # 5. Remove any .gitattributes or compatibility reports (not in official format)
    
    # 6. List all created files in official format
    print_official_hf_files(save_path)
    
    return save_path

def create_generation_config_hf(save_path, config):
    """Create generation_config.json for Hugging Face compatibility."""
    
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": getattr(config, 'pad_token_id', None),
        "transformers_version": "4.36.0",
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "force_words_ids": None,
        "renormalize_logits": False,
        "constraints": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None
    }
    
    with open(save_path / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)


def save_tokenizer_official_format(tokenizer, save_path):
    """Save tokenizer in the official HF format (without tokenizer.model)."""
    
    # Save tokenizer normally first
    tokenizer.save_pretrained(save_path)
    
    # Ensure we have the required files
    required_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for file in required_files:
        if not (save_path / file).exists():
            print(f"  Warning: {file} not found")


def print_official_hf_files(save_path):
    """Print files in the exact official Hugging Face format."""
    
    print(f"\n📁 Official Hugging Face Model Files:")
    print("=" * 70)
    
    # Expected files in exact order
    expected_files = [
        "config.json",
        "generation_config.json", 
        "README.md",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    # Find all safetensors files
    safetensors_files = sorted([f for f in save_path.glob("*.safetensors") if not f.name.endswith('.index.json')])
    safetensors_index = save_path / "model.safetensors.index.json"
    
    def print_file_info(filename, category_emoji=""):
        file_path = save_path / filename
        if file_path.exists():
            size = file_path.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB" 
            else:
                size_str = f"{size} B"
            print(f"{category_emoji} {filename:<35} ({size_str})")
            return True
        else:
            print(f"❌ {filename:<35} (missing)")
            return False
    
    # Print config files
    print("\n📋 Configuration Files:")
    for file in ["config.json", "generation_config.json"]:
        print_file_info(file, "✓")
    
    # Print model files
    print("\n🤖 Model Files:")
    if safetensors_index.exists():
        print_file_info("model.safetensors.index.json", "✓")
    
    # Print sharded model files
    for i, shard_file in enumerate(safetensors_files, 1):
        print_file_info(shard_file.name, "✓")
    
    # Print tokenizer files  
    print("\n📝 Tokenizer Files:")
    tokenizer_files = ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    for file in tokenizer_files:
        print_file_info(file, "✓")
    
    # Print documentation
    print("\n📚 Documentation:")
    print_file_info("README.md", "✓")
    
    # Summary
    total_files = len(expected_files) + len(safetensors_files) + (1 if safetensors_index.exists() else 0)
    print(f"\n" + "=" * 70)
    print(f"🎉 Total files: {total_files}")
    print(f"📦 Model shards: {len(safetensors_files)}")
    print(f"💾 Format: Sharded SafeTensors (Official HF Format)")
    print(f"🚀 Ready for Hugging Face Hub!")
    
    return total_files


def create_matched_llama_model(tokenizer_path_or_name, model_config=None, save_path=None):
    """
    Create a Llama model that perfectly matches an existing tokenizer.
    
    Args:
        tokenizer_path_or_name: Path or name of tokenizer to match
        model_config: Optional custom model configuration
        save_path: Optional path to save the matched model
    
    Returns:
        Tuple of (model, config, tokenizer)
    """
    
    print(f"Creating model to match tokenizer: {tokenizer_path_or_name}")
    
    # 1. Get tokenizer information
    tokenizer_info, tokenizer = get_tokenizer_info(tokenizer_path_or_name)
    if tokenizer_info is None:
        raise ValueError(f"Could not load tokenizer from {tokenizer_path_or_name}")
    
    # 2. Create matching config
    config = create_matched_model_config(tokenizer_info, model_config)
    print(f"\nCreated config with vocab_size: {config.vocab_size}")
    
    # 3. Initialize model
    print("Initializing model...")
    model = LlamaForCausalLM(config)
    
    # 4. Apply weight initialization (from previous code)
    model.apply(init_llama_weights)

    # Print model info
    param_info = get_parameter_count(model)
    print(f"Model initialized with {param_info['total_parameters_millions']:.2f}M parameters")
    
    # Special initialization for output projection
    # Scale down the output layer weights for better training stability
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        std = 1.0 / math.sqrt(config.hidden_size)
        torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=std)
    
    # 5. Initialize special tokens properly
    # initialize_special_tokens(model, tokenizer_info)
    
    # 6. Verify model and tokenizer compatibility
    verify_model_tokenizer_compatibility(model, tokenizer)
    
    # 7. Save if requested
    if save_path:
        save_matched_model_huggingface_format(model, tokenizer, save_path)
    
    return model, config, tokenizer

if __name__ == "__main__":
    setup_debugpy(force=True)
    try:
        model, config, tokenizer = create_matched_llama_model(
            tokenizer_path_or_name="/home/aiops/zhangfz/pretrained_models/Llama-2-7b-hf",
            save_path="/home/aiops/zhangfz/LLaMA-Factory-Muon/running_ckpts/initializations/llama-2-100M",
            model_config = {
            'vocab_size': 32000,
            'hidden_size': 768,
            'intermediate_size': 4*768,
            'num_hidden_layers': 12,
            'num_attention_heads': 6,
            'num_key_value_heads': 6,  # For grouped-query attention
            'hidden_act': "silu",
            'max_position_embeddings': 2048,
            'initializer_range': 0.02,
            'rms_norm_eps': 1e-6,
            'rope_theta': 10000.0,
        }
        )
        print("✓ Successfully created model matching Llama-2 7B tokenizer")
        print(tokenizer.chat_template)
        
    except Exception as e:
        print(f"Could not load Llama-2 tokenizer: {e}")
        print("Trying with a different approach...")
        
        
        # You would replace this with your actual tokenizer path
        print("Please provide a valid tokenizer path or model name to continue")

# Example usage
# if __name__ == "__main__":
#     setup_debugpy(force=True)
#     # Initialize a small Llama model for testing
#     model = initialize_llama_model({
#         'vocab_size': 32000,
#         'hidden_size': 768,
#         'intermediate_size': 4*768,
#         'num_hidden_layers': 12,
#         'num_attention_heads': 6,
#         'num_key_value_heads': 6,  # For grouped-query attention
#         'hidden_act': "silu",
#         'max_position_embeddings': 2048,
#         'initializer_range': 0.02,
#         'rms_norm_eps': 1e-6,
#         'rope_theta': 10000.0,
#     })
    
#     # Print model info
#     param_info = get_parameter_count(model)
#     print(f"Model initialized with {param_info['total_parameters_millions']:.2f}M parameters")
    
#     # Test forward pass with random input
#     batch_size, seq_len = 2, 128
#     input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
#     with torch.no_grad():
#         outputs = model(input_ids)
#         print(f"Output logits shape: {outputs.logits.shape}")
#         print("Model initialization successful!")