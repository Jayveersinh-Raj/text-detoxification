# Jayveersinh Raj
# BS20-DS01
# j.raj@innopolis.university

## To get the interim data from raw data
First, install the requirements in the environment, then navigate to`src/data/make_dataset.py`.
    
    python3 make_dataset.py
    
The interim dataset would be stored in the `data/interim` directory in the root folder, which would be used for training the model, from this dataset itself test set would be split with a seed of `42`.
## To train the final solution model (bloom-3b-8bit-quantized-lora-adapter):
First, install the requirements in the environment, then navigate to`src/models/train_model.py`.

    python3 train_model.py

The trained checkpoints will be stored in the `models/bloom-detoxification/checkpoint-800/` of the root folder, the repository already has trained checpoints, if its
trained again the checkpoints will be updated.

## To use and check the model for inference (bloom-3b-8bit-quantized-lora-adapter):
Navigate to`src/models/predict_model.py`

    python3 predict_model.py

You'll be asked for input of the toxic text, and the output would be both the toxic text and the corresponding result.

## Example
<img width="400" alt="image" src="https://github.com/Jayveersinh-Raj/text-detoxification/assets/69463767/05aff7fb-66c7-4db6-8705-b7630caf3a3f">


## Note:
### It requires a GPU of atleast 16 GBs (free version of google colab would help) because of a 3 billion parameterized base large language model.
    

