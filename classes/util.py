# for helper utils
import os
import pandas as pd
import pickle
import re
import json
import networkx as nx
import logging
from datetime import datetime
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# for loading models
import torch
# from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    pipeline
)
from constants import MODELS_PATH, MODEL_ID, MAX_NEW_TOKENS, MODEL_BASENAME
from langchain.llms import HuggingFacePipeline

class InfoDot:

    next_id = 1
    def __init__(self, name, info, dot_type="person", aliases=[], doc=None, id=None, db_id=None):
        self.name = name
        self.aliases = []
        if aliases:
            self.aliases.extend(aliases)
        # info is both dot and synthesized info, need to update when adding new one
        self.info = info
        # doc can be None, for IT type dots
        self.doc = doc
        # assign unique id to this dot
        if not id:
            self.id = Dot.next_id
            InfoDot.next_id += 1
        else:
            self.id = id
        self.dot_type = dot_type
        if self.name == "event":
            self.dot_type = "event"
        self.subsumed = False
        # saving the database id returned from add_documents

    # return the content of this dots for printing and logging
    # can also create a dictionary and return that, default
    def __str__(self):
        # return f"<Dot:{self.info}; From Doc:{self.doc}; Belong to IT:{self.ITlist}>"
        return str({'id':self.id, 'name':self.name, 'aliases':str(self.aliases), 'info':self.info, 'doc':self.doc})
    
    def __lt__(self, obj):
        return ((self.id) < (obj.id))
    
    # return the content of this dots for printing and logging
    # can also create a dictionary and return that, default
    def get_dict(self):
        # return f"<Dot:{self.info}; From Doc:{self.doc}; Belong to IT:{self.ITlist}>"
        return {'id':self.id, 'name':self.name, 'aliases': [x for x in self.aliases], 'info':self.info, 'doc':self.doc}

    def equals(self, dot):
        return (dot.name == self.name) and (dot.doc == self.doc)

    
def update_info_for_persons(node, new_dot, llm):
    open_ai_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "The following two reports are on a same person. Condense them into one paragraph. You must keep all information intact."),
            ("human", "Report1: {document1}\nReport2: {document2}")
        ],
    )
    llama_template = "<s>[INST] <<SYS>>\nThe following two reports are on a same person. Condense them into one paragraph. You must keep all information intact.\n<</SYS>>\n\nReport1: {document1}\nReport2: {document2}\n\nCondense them into one paragraph. You must keep all information intact.\n\nAnswer:\n[/INST]"
    llama_prompt = PromptTemplate(
        input_variables=["document1", "document2"],
        template=llama_template,
    )

    if llm['openAI']:
        prompt_text = open_ai_prompt.format_messages(document1 = node.info, document2 = new_dot['dot'])
        try:
            return llm['openAI']['gpt35'].invoke(prompt_text).content.strip()
        except Exception as e:
            logging.error(f"Error: {e} occured in update_info_for_persons")
            if llm['local_llm']:
                logging.info(f"With Llama: running update_info_for_persons")
                prompt_text = llama_prompt.format(document1 = node.info, document2 = new_dot['dot'])
                return llm['local_llm'].invoke(prompt_text).strip()
            else:
                raise e
    elif llm['local_llm']:
        logging.info(f"With Llama only mode: running update_info_for_persons")
        prompt_text = llama_prompt.format(document1 = node.info, document2 = new_dot['dot'])
        return llm['local_llm'].invoke(prompt_text).strip()
    else:
        logging.error(f"Error: No LLM found in update_info_for_persons")
        return None


def which_person(dot, llm):
    open_ai_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Read the report and output the name and aliases of the person the report is talking about in this format: #name: name #aliases: alias1, alias2, ... If this is an event-based report, set the name as 'event'."),
            ("human", "Report: {document1}")
        ]
    )
    llama_template = "<s>[INST] <<SYS>>\nRead the report and output the name and aliases of the person the report is talking about in this format: #name: name #aliases: alias1, alias2, ... If this is an event-based report, set the name as 'event'.\n<</SYS>>\n\nReport: {document1}\n\nAnswer:\n[/INST]"
    llama_prompt = PromptTemplate(
        input_variables=["document1"],
        template=llama_template,
    )

    if isinstance(dot, str):
        question = dot.strip()
    else:
        question = dot.info.strip()
 
    # We will do this with llama only for now
    if llm['openAI']:
        prompt_text = open_ai_prompt.format_messages(document1 = question)
        try:
            return llm['openAI']['gpt35'].invoke(prompt_text).content.strip()
        except Exception as e:
            logging.error(f"Error: {e} occured in entity_extraction for dot {question}. ")
            if llm['local_llm']:
                logging.info(f"With Llama: running entity_extraction")
                prompt_text = llama_prompt.format(document1 = question)
                return llm['local_llm'].invoke(prompt_text).strip()
            else:
                raise e
    elif llm['local_llm']:
        logging.info(f"With Llama only mode: running entity_extraction")
        prompt_text = llama_prompt.format(document1 = question)
        return llm['local_llm'].invoke(prompt_text).strip()
    else:
        logging.error(f"Error: No LLM found to extract entities for dot {question}")
        return None


class Entity(BaseModel):
    """Output the LLM input with the entities in the search query."""
    entities: str = Field(description=("All entities without commas"))

def extract_entities(dot, llm):
    # fixing prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at extracting entities from the sentence. All entities must be from the sentence."),
            ("human", "{question}"),
        ]
    )
    if isinstance(dot, str):
        question = dot.strip()
    else:
        question = dot.info.strip()

    # We will do this with llama only for now
    if llm['openAI']:
        structured_llm = llm['openAI'].with_structured_output(Entity)
        query_analyzer = prompt | structured_llm
        try:
            return query_analyzer.invoke({"question": question}).entities.strip()
        except Exception as e:
            logging.error(f"Error: {e} occured in entity_extraction for dot {dot}")
            return dot
    else:
        return dot


# also supports taking in a list_of_dots for pickling, not changing the argument's name though
def save_load_memory_stream(file_name, mode="save", memory_stream=None):
    """mention mode "save" or "load". It will only work with the list of dots"""
    if "json" not in file_name:
        if mode=="save":
            with open(file_name, "wb") as fp:   #Pickling
                if isinstance(memory_stream, list):
                    pickle.dump(memory_stream, fp)
                else:
                    pickle.dump(memory_stream.dots, fp)
        else:
            with open(file_name, "rb") as fp:   # Unpickling
                return pickle.load(fp)
    elif "json" in file_name:
        if mode=="save":
            with open(file_name, 'w', encoding='utf-8') as fp:
                json.dump(memory_stream.snapshot(), fp, ensure_ascii=False, indent=4)
        else:
            with open(file_name, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                return data



def load_quantized_model_qptq(model_id, model_basename, device_type, logging):
    logging.info("Using AutoGPTQForCausalLM for quantized models")

    if ".safetensors" in model_basename:
        # Remove the ".safetensors" ending if present
        model_basename = model_basename.replace(".safetensors", "")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    logging.info("Tokenizer loaded")

    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device_map="auto",
        use_triton=False,
        quantize_config=None,
    )
    return model, tokenizer


def load_full_model(model_id, model_basename, device_type, logging):
    if device_type.lower() in ["mps", "cpu"]:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir="./models/")
        model = LlamaForCausalLM.from_pretrained(model_id, cache_dir="./models/")
    else:
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
        logging.info("Tokenizer loaded")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            # trust_remote_code=True, # set these if you are using NVIDIA GPU
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    return model, tokenizer


def load_model(model_type="both", device_type="cuda", model_id=MODEL_ID, model_basename=None, LOGGING=logging, openai_temperature=0.2, local_temperature=0.7):
    logging.info(f"Loading model_type: {model_type}")
    local_llm = None
    open_ai_llm_dict = None

    if model_type in ["llama", "both"]:
        if model_basename is not None:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=MAX_NEW_TOKENS,
            temperature=local_temperature,
            top_p=0.1,
            top_k=40,
            repetition_penalty=1.176,
            generation_config=generation_config,
            truncation=True
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        logging.info(f"Local LLM Loaded with model_type: {model_type}")
        
    if model_type in ["openAI", "both"]:
        llm_gpt4 = AzureChatOpenAI(
            temperature=openai_temperature,
            api_key=os.getenv("GPT4o_KEY"),
            api_version=os.getenv("GPT4o_API_VERSION"),
            azure_deployment=os.getenv("GPT4o_DEPLOYMENT"),
            azure_endpoint=os.getenv("GPT4o_ENDPOINT"),
        )
        logging.info(f"OpenAI model Loaded with model_type: {model_type}")

        open_ai_llm_dict = {"gpt35":None, "gpt4":llm_gpt4}
    
    return {"local_llm":local_llm, "openAI":open_ai_llm_dict}