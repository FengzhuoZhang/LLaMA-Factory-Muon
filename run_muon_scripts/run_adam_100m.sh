LLAMA2_HF_PATH="./"
python train_muon/train_adam_100m.py  --pretrained_model_path LLAMA2_HF_PATH  --seed 42 --base_yaml_file "examples/train_full/llama3_full_pt.yaml"