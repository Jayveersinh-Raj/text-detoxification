import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Checkpoints path, and loading the base model
peft_model_id = "../../models/bloom-detoxification/chekpoint-800"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
qa_model = PeftModel.from_pretrained(model, peft_model_id)


def make_inference(toxic: str) -> str:
  """
  Function to make inference on toxic sentence

  Parameters:
  -----------
  toxic (str): The toxic sentence to detoxify

  Returns:
  ----------
  Detoxified sentence
  """

  batch = tokenizer(f"### Detoxify:\n{toxic}\n\n### detoxified:\n", return_tensors='pt').to("cuda")
  with torch.cuda.amp.autocast():
    output_tokens = qa_model.generate(**batch, max_new_tokens=200)

  return tokenizer.decode(output_tokens[0], skip_special_tokens=True).split("\n")[-1]

text = input("Enter the text to detoxify: ")
pred = make_inference(toxic=text)
print(f"The toxic sentence: {text}\n Detoxified: {pred}")

