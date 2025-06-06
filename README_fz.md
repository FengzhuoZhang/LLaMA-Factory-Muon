To run this repo, you need to:

for all the running below, you need to modify the path to your own path

Step 1:
    config your environ according to README.md

Step 2:
    download llama2 via:

    ```bash
    git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
    ```

Step 3:
    download fineweb data by running:

    ```bash
    python dataset_processing/download.py
    ```

    create the data_info.json for it by running: 

    ```bash
    python dataset_processing/create_data_info.py
    ```

Step 4:
    train model by running: 

    ```bash
    bash run_muon_scripts/run_adam_100m.sh
    ```



