import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers


# Putting CUDA device to environment
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Loading model checkpoints, and tokenizer of the base model
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-3b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")



# Parameter efficient (freezing layers)
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)



# Trainable parameters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


    
# Obtaining lora model (lora config)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)



# Load dataset
dataset = load_dataset("csv", data_files="../../data/interim/detoxification_dataset.csv", split = "train[400000:500000]")

# Train test split
data = dataset.train_test_split(test_size=0.01)



# Prompt template for the decoder model
def create_prompt(toxic: str, detoxified: str) -> str:
  """
  Creates the prompt template for the decoder model with toxic, and detoxified sentences

  Parameters:
  -----------
  toxic (str): The toxic text
  detoxified (str): The corresponding detoxified text

  Returns:
  ----------
  The entire prompt with toxic, and detoxified texts
  """

  if len(detoxified) < 1:
    detoxified = "Cannot Find Answer"
  else:
    detoxified = detoxified
  prompt_template = f"### Detoxify:\n{toxic}\n\n### detoxified:\n{detoxified}</s>"
  return prompt_template

mapped_data_train = data["train"].map(lambda samples: tokenizer(create_prompt(samples['toxic'], samples['detoxified'])))
mapped_data_test = data["test"].map(lambda samples: tokenizer(create_prompt(samples['toxic'], samples['detoxified'])))



# Training LoRA model
trainer = transformers.Trainer(
    model=model,
    train_dataset=mapped_data_train,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=800,
        learning_rate=1e-3,
        fp16=True,
        logging_steps=100,
        save_steps=200,
        overwrite_output_dir=True,
        save_total_limit=3,
        output_dir='../../models/bloom-detoxification',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()