import torch
from transformers import BitsAndBytesConfig
import json

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
from transformers import AutoModelForCausalLM, LlamaTokenizer, pipeline
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "LlamaTurk-7b-i-checkpoints"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
				    quantization_config=nf4_config, 
					    torch_dtype=torch.bfloat16,
                                             device_map='auto') 

sampling_params = dict(do_sample=True, temperature=0.3, top_k=50, top_p=0.9)

pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer,
                device_map=device,
                max_new_tokens=1024, 
                return_full_text=True,
                repetition_penalty=1.1
               )

DEFAULT_SYSTEM_PROMPT = "Lütfen verilen soru ve doğru cevabı için 5 doğru ve 5 yanlış cümle üretiniz.\nCikti formati = ['Dogru'=[], 'Yanlis'=[]]\n"

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

def generate_prompt(sentence, answer, system_prompt=DEFAULT_SYSTEM_PROMPT):
    instruction = f"[Given Sentence Start]\n{sentence}\n[Given Sentence End]\n" +\
                  f"[Given Answer Start]\n{answer}\n[Given Answer End]\n\n"
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

def generate_output(user_query, sys_prompt=DEFAULT_SYSTEM_PROMPT):
    prompt = generate_prompt(user_query, sys_prompt)
    outputs = pipe(prompt,
               **sampling_params
              )
    return outputs[0]["generated_text"].split("[/INST]")[-1]
with open('train-v0.1.json', "r", encoding='utf8') as file:
    data = json.load(file)

# extract questions and answers from dataset
questions_answers = []
for item in data['data']:
    for element in item['paragraphs']:
        if len(element['qas'])>0:
            question = element['qas'][0]['question']
            answer = element['qas'][0]['answers'][0]['text']
            questions_answers.append({'question': question, 'answer': answer})

generation = []
for item in questions_answers[0:500]:
    question = item['question']
    answer = item['answer']
    prompt = generate_output(question, answer)
    generation.append(prompt)

with open('llama_trendyol_generation.json', 'w', encoding='utf8') as file:
    json.dump(generation, file)
