from flask import Flask, request, jsonify
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, GenerationConfig
from peft import PeftModel, LoraConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch

app = Flask(__name__)

# Model and tokenizer setup with LangChain
def create_model_and_tokenizer():
    model_name = "NousResearch/llama-2-7b-chat-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Assuming PeftModel and additional configurations are correctly set up for your use case
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load the model with your specific configurations if needed
    # This step may need to be adjusted based on your model's setup
    model = PeftModel.from_pretrained(model, 'NousResearch/llama-2-7b-chat-hf')
    
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    
    return llm

llm = create_model_and_tokenizer()

# Define the API endpoint for asking questions
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        result = llm(question)
        answer = result
        print(answer)
        
        return jsonify({'answer': answer}), 200
    except Exception as e:
        print("Unexpected server error:", e)
        traceback.print_exc()
        return jsonify({'error': 'Server encountered an unexpected error'}), 500

if __name__ == '__main__':
    app.run(debug=True,host='10.100.102.6', port=8080)  
