import re
import os
import torch
import pandas as pd
import json
import argparse
import nltk
import requests
from rank_bm25 import BM25Okapi
import openai
import spacy
from openai import OpenAI
from openai import AzureOpenAI
from transformers import set_seed, BitsAndBytesConfig, AutoModelForCausalLM, LlamaTokenizer, pipeline
import boto3
from botocore.exceptions import ClientError

class ModelGenerator:
    def __init__(self, api_endpoint, boto3_config, openai_config):
        self.function_name = 'call_claude_service'
        self.function_url = f'{api_endpoint}/{self.function_name}'
        self.boto3_client = self.get_boto3_client(boto3_config)
        self.openai_client = self.get_openai_client(openai_config)
        self.openai_deployment_name = openai_config['deployment_name']

        # Trendyol LLM setup
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "Trendyol-LLM-7b-chat-v0.1"
        self.tokenizer = LlamaTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         quantization_config=nf4_config, 
                                                         torch_dtype=torch.bfloat16,
                                                         device_map='auto')
        self.sampling_params = dict(do_sample=True, temperature=0.3, top_k=50, top_p=0.9)
        self.pipe = pipeline("text-generation", 
                             model=self.model, 
                             tokenizer=self.tokenizer,
                             device_map=self.device,
                             max_new_tokens=1024, 
                             return_full_text=True,
                             repetition_penalty=1.1)
        self.DEFAULT_SYSTEM_PROMPT = "Lütfen verilen soru ve doğru cevabı için 5 doğru ve 5 yanlış cümle üretiniz\n"
        self.TEMPLATE = (
            "[INST] <<SYS>>\n"
            "{system_prompt}\n"
            "<</SYS>>\n\n"
            "{instruction} [/INST]"
        )

    def get_boto3_client(self, config):
        return boto3.client(
            service_name=config['aws_service_name'],
            region_name=config['aws_region_name'],
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key']
        )

    def get_openai_client(self, config):
        return AzureOpenAI(
            api_key=config['api_key'],
            api_version=config['api_version'],
            azure_endpoint=config['azure_endpoint']
        )

    def llama_get_prompt(self, sentence, answer):
        instruction = (
            f"Lütfen verilen soru ve doğru cevabı için 5 doğru ve 5 yanlış cümle üretiniz"
            f"[Given Sentence Start]\n{sentence}\n[Given Sentence End]\n"
            f"[Given Answer Start]\n{answer}\n[Given Answer End]\n\n"
        )
        return f"user\n{instruction}\n\nassistant"

    def llama_get_response(self, prompt):
        native_request = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
        }
        request = json.dumps(native_request)
        try:
            response = self.boto3_client.invoke_model(modelId="meta.llama3-1-70b-instruct-v1:0", body=request)
            model_response = json.loads(response["body"].read())
            return model_response["generation"]
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke LLaMA model. Reason: {e}")
            exit(1)

    def claude_get_prompt(self, sentence, answer):
        instruction = (
            f"Lütfen verilen soru ve doğru cevabı için 5 doğru ve 5 yanlış cümle üretiniz"
            f"[Given Sentence Start]\n{sentence}\n[Given Sentence End]\n"
            f"[Given Answer Start]\n{answer}\n[Given Answer End]\n\n"
        )
        return f"Human: {instruction}\n\nAssistant:"

    def claude_get_response(self, prompt, version='v1'):
        model_id = 'anthropic.claude-instant-v1' if version == 'v1' else 'anthropic.claude-v2'
        payload = {
            'modelID': model_id,
            'full_prompt': prompt,
            'temperature': 0.9,
            'top_p': 0.95
        }
        response = requests.post(self.function_url, data=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")

    def openai_get_prompt(self, sentence, answer):
        return (
            f"Please generate 5 correct and 5 wrong sentences for given sentence and its correct answer"
            f"[Given Sentence Start]\n{sentence}\n[Given Sentence End]\n"
            f"[Given Answer Start]\n{answer}\n[Given Answer End]\n\n"
        )

    def openai_generate_response(self, prompt):
        response = self.openai_client.chat.completions.create(
            model=self.openai_deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response.choices[0].message.content

    def generate_prompt_trendyol(self, sentence, answer, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        instruction = (
            f"[Given Sentence Start]\n{sentence}\n[Given Sentence End]\n"
            f"[Given Answer Start]\n{answer}\n[Given Answer End]\n\n"
        )
        return self.TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})

    def generate_output_trendyol(self, user_query, sys_prompt=None):
        prompt = self.generate_prompt_trendyol(user_query, sys_prompt)
        outputs = self.pipe(prompt, **self.sampling_params)
        return outputs[0]["generated_text"].split("[/INST]")[-1]

    def generate_responses(self, questions_answers, model):
        generation = []
        for item in questions_answers:
            question = item['question']
            answer = item['answer']
            if model == 'llama':
                prompt = self.llama_get_prompt(question, answer)
                response = self.llama_get_response(prompt)
            elif model == 'claude':
                prompt = self.claude_get_prompt(question, answer)
                response = self.claude_get_response(prompt)
            elif model == 'openai':
                prompt = self.openai_get_prompt(question, answer)
                response = self.openai_generate_response(prompt)
            elif model == 'trendyol':
                response = self.generate_output_trendyol(question, answer)
            generation.append(response)
        return generation

    def save_generation_to_file(self, generation, filename):
        with open(filename, 'w', encoding='utf8') as file:
            json.dump(generation, file)


if __name__ == '__main__':
    api_endpoint = API_ENDPOINT
    boto3_config = {
        "aws_service_name": 'bedrock-runtime',
        "aws_region_name": 'us-west-2',
        "aws_access_key_id": 'YOUR_AWS_ACCESS_KEY_ID',
        "aws_secret_access_key": 'YOUR_AWS_SECRET_ACCESS_KEY',
        "model_id": "meta.llama3-1-70b-instruct-v1:0",
        "max_tokens_to_sample": 10_000,
        "temperature": 0.0,
        "top_p": 0.95
    }
    openai_config = {
        "api_key": "YOUR_API_KEY",
        "api_version": "2024-02-01",
        "azure_endpoint": "https://YOUR_OPENAI_ENDPOINT",
        "deployment_name": 'omni'
    }

    with open('train-v0.1.json', "r", encoding='utf8') as file:
        data = json.load(file)

    # Extract questions and answers from dataset
    questions_answers = []
    for item in data['data']:
        for element in item['paragraphs']:
            if len(element['qas']) > 0:
                question = element['qas'][0]['question']
                answer = element['qas'][0]['answers'][0]['text']
                questions_answers.append({'question': question, 'answer': answer})

    model_generator = ModelGenerator(api_endpoint, boto3_config, openai_config)

    # Generate and save Trendyol responses
    trendyol_generation = model_generator.generate_responses(questions_answers, 'trendyol')
    model_generator.save_generation_to_file(trendyol_generation, 'llama_trendyol_generation.json')

    # Generate and save LLaMA responses
    llama_generation = model_generator.generate_responses(questions_answers, 'llama')
    model_generator.save_generation_to_file(llama_generation, 'llama3_generation.json')

    # Generate and save GPT-4 responses
    gpt4_generation = model_generator.generate_responses(questions_answers, 'openai')
    model_generator.save_generation_to_file(gpt4_generation, 'gpt4_generation.json')

    # Generate and save Claude responses
    claude_generation = model_generator.generate_responses(questions_answers, 'claude')
    model_generator.save_generation_to_file(claude_generation, 'claude_generation.json')
