import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.create_model import create_matched_llama_model
import argparse
import yaml
from llamafactory.cli import main


parser = argparse.ArgumentParser(description="Model Training with Adam")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--pretrained_model_path", type=str, default="/home/aiops/zhangfz/pretrained_models/Llama-2-7b-hf")
parser.add_argument("--base_yaml_file", type=str, default="/home/aiops/zhangfz/LLaMA-Factory-Muon/examples/train_full/llama3_full_pt.yaml")
exp_args = parser.parse_args()

exp_name = "adam_seed_"+str(exp_args.seed)+"_100M"
initial_ckpt_folder = f"/home/aiops/zhangfz/LLaMA-Factory-Muon/running_ckpts/initializations/"+exp_name
log_folder = f"/home/aiops/zhangfz/LLaMA-Factory-Muon/running_logs/"+exp_name
os.makedirs(log_folder, exist_ok=True)

if not os.path.exists(initial_ckpt_folder):
    model, config, tokenizer = create_matched_llama_model(
                tokenizer_path_or_name= exp_args.pretrained_model_path,
                save_path=initial_ckpt_folder,
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
    
print("======================================================================")
print("Initialized Model!")
print("======================================================================")

### load and modify base yaml
with open(exp_args.base_yaml_file, 'r') as file:
    data = yaml.safe_load(file)

data["model_name_or_path"] = initial_ckpt_folder
data["output_dir"] = log_folder
data["seed"] = exp_args.seed
data["data_seed"] = exp_args.seed

dataset_folder = "/home/aiops/zhangfz/LLaMA-Factory-Muon/dataset_self/fineweb/"
data["dataset_dir"] = dataset_folder
data_paths = []
for i in range(0,10):
    data_paths.append("fineweb_"+str(i))
data_str = ", ".join(data_paths)
# print(data_str)
data["dataset"] = data_str

with open(log_folder+'/'+exp_name +'.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

# print(log_folder+'/'+exp_name +'.yaml')
print("======================================================================")
print("Read the Yaml Config!")
print("======================================================================")


### begin training
import subprocess

print("======================================================================")
print("Start Training!")
print("======================================================================")

result = subprocess.run([
    "llamafactory-cli", "train", 
    log_folder+'/'+exp_name +'.yaml'
])
