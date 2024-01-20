import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import set_seed, MixtralForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from os import environ

BASE_MODEL_PATH = environ.get("BASE_MODEL_PATH")
HF_MODEL_EXPORT_PATH = environ.get("HF_MODEL_EXPORT_PATH")
ONNX_MODEL_EXPORT_PATH = environ.get("ONNX_MODEL_EXPORT_PATH")

def get_model(export = False):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True
    )
    model = MixtralForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH, 
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto",
        export=export
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    return model, tokenizer

def get_onnx_model_format():
    ORTModelForSequenceClassification.from_pretrained(HF_MODEL_EXPORT_PATH).save_pretrained(ONNX_MODEL_EXPORT_PATH)
    AutoTokenizer.from_pretrained(HF_MODEL_EXPORT_PATH).save_pretrained(ONNX_MODEL_EXPORT_PATH)
    
if __name__ == "__main__":
    [model, tokenizer] = get_model(export=True)
    
    set_seed(0)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a thug",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
    
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_length = model_inputs.shape[1]
    generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=1024)
    print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
    
    model.save_pretrained(HF_MODEL_EXPORT_PATH)
    tokenizer.save_pretrained(HF_MODEL_EXPORT_PATH)
    
    get_onnx_model_format()