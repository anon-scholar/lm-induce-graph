# for helper utils
import os
import pandas as pd
import pickle
import re
import numpy as np
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

import re

def extract_connections_pair(text):
    # Initialize the dictionary to store connections
    connections = {}

    # Regular expression patterns to match various connection formats
    patterns = [
        r'(?:Pair|Connection)\s*(?:\d+\.?|:)?\s*(.*?)\s+and\s+(.*?)\s*[:|-]\s*(.+)',  # Matches "Pair 1. A and B: explanation" or "Connection: A and B - explanation"
        r'(\w+(?:\s+\w+)?)\s+is connected to\s+(\w+(?:\s+\w+)?)\s*[:|-]\s*(.+)',  # Matches "A is connected to B: explanation"
        r'Connection between\s+(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)\s*[:|-]\s*(.+)'  # Matches "Connection between A and B: explanation"
    ]
    anti_patterns = r'[.\s]*(.+)No direct connection[.\s]*(.+)'

    # Iterate through the lines
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if the line matches any of the patterns
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            anti_match = re.match(anti_patterns, line, re.IGNORECASE)
            # print("anti match: ", anti_match)
            if match and not anti_match:
                person1, person2, explanation = match.groups()
                key = f"{person1.strip()} - {person2.strip()}"
                connections[key] = explanation.strip()
                break

    # Check if no connections were found
    if not connections:
        no_connection_patterns = [
            r'(?:#Connections:|No connections found)[.\s]*(.+)',
            r'Unable to identify any connections[.\s]*(.+)',
            r'No (?:clear|obvious|apparent) connections (?:found|identified)[.\s]*(.+)'
        ]
        for pattern in no_connection_patterns:
            no_connection_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if no_connection_match:
                connections['No connections'] = no_connection_match.group(1).strip()
                break

    return connections


def get_id_from_name(extracted_content, name_id_dict):
  predicted_connections = []
  for pair in extracted_content.keys():
    person1 = pair.split(' - ')[0]
    if person1 in name_id_dict.keys():
      person1_id = name_id_dict[person1]
    else:
      person1_id = 'X'
    person2 = pair.split(' - ')[1]
    if person2 in name_id_dict.keys():
      person2_id = name_id_dict[person2]
    else:
      person2_id = 'X'
    if person1_id == 'X' or person2_id == 'X':
      continue
    else:
      predicted_connections.append(set([person1_id, person2_id]))

  return predicted_connections



#Create adjacency matrix from lists of connected and disconnected noded
def generate_adjacency_matrix(all_ids, connected_pairs):
    id_to_index = {id: index for index, id in enumerate(all_ids)}
    total_people = len(all_ids)
    adj_matrix = np.zeros((total_people, total_people), dtype=int)

    for person1, person2 in connected_pairs:
        if person1 in id_to_index and person2 in id_to_index:
            index1 = id_to_index[person1]
            index2 = id_to_index[person2]

            adj_matrix[index1][index2] = 1
            adj_matrix[index2][index1] = 1
        else:
            print(f"Warning: ID {person1} or {person2} not found in the list of all IDs.")

    return adj_matrix


def calculate_score_v4(true_adjacency_matrix, predicted_adjacency_matrix):
    if not isinstance(true_adjacency_matrix, np.ndarray):
        true_adjacency_matrix = np.array(true_adjacency_matrix)
    if not isinstance(predicted_adjacency_matrix, np.ndarray):
        predicted_adjacency_matrix = np.array(predicted_adjacency_matrix)

    n = true_adjacency_matrix.shape[0]

    # Create mask for upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Apply mask to get only upper triangle values
    true_upper = true_adjacency_matrix[mask]
    pred_upper = predicted_adjacency_matrix[mask]

    # Calculate base metrics
    TP = np.sum((true_upper == 1) & (pred_upper == 1))
    FP = np.sum((true_upper == 0) & (pred_upper == 1))
    FN = np.sum((true_upper == 1) & (pred_upper == 0))

    # Total actual positive connections
    total_positives = np.sum(true_upper == 1)

    # Normalize only with total positives (to avoid sensitivity to extra entities)
    normalization_factor = 100 / (2 * total_positives) if total_positives > 0 else 1

    # Compute score without TN to be noise-agnostic
    score = (2 * TP - 0.5 * FP - 1.0 * FN) * normalization_factor
    score = max(0, score)  # Ensure non-negative scores

    # Precision, recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Store metrics
    metrics = {
        'true_positives': int(TP),
        'false_positives': int(FP),
        'false_negatives': int(FN),
        'total_actual_positives': int(total_positives),
        'precision': precision,
        'recall': recall,
        'F1_score': f1_score,
        'weights': {
            'TP_weight': 2 * normalization_factor,
            'FP_weight': -0.5 * normalization_factor,
            'FN_weight': -1.0 * normalization_factor
        }
    }

    return score, metrics




def gpt_connection(llm, all_docs):
    system_message = """You are an FBI agent, working with dossiers of multiple persons. Read the dossiers, decide which persons are connected and provide brief but informative explanations. Only list the pairs with direct connections. Follow this format strictly:

    # Connections: number of connections found

    Pair 1. Person A and Person B: [Brief explanation of their connection]
    Pair 2. Person A and Person C: [Brief explanation of their connection]
    Pair 3. Person D and Person E: [Brief explanation of their connection]
    ...

    Base your analysis solely on the provided information. Do not invent details. If no connections are found, state "No connections found" after # Connections: and explain why. Follow the format strictly.
    """

    connection_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "Dossiers:\n{all_docs}\n")
        ],
    )
    messages = connection_prompt_template.format_messages(all_docs=all_docs)
    # number_of_tokens = llm['openAI']['gpt4'].get_num_tokens_from_messages(messages)
    # print(number_of_tokens)

    #calling the prompt
    resp = llm['openAI']['gpt4'].invoke(messages).content.strip()
    # number_of_tokens = llm_gpt35['openAI'].get_num_tokens_from_messages(resp)
    return resp


def gpt_connection_cot_basic(llm, all_docs):
    system_message = """You are an FBI agent, working with dossiers of multiple persons. Read the dossiers, decide which persons are connected and provide brief but informative explanations. Only list the pairs with direct connections. Follow this format strictly:

    # Connections: number of connections found

    Pair 1. Person A and Person B: [Brief explanation of their connection]
    Pair 2. Person A and Person C: [Brief explanation of their connection]
    Pair 3. Person D and Person E: [Brief explanation of their connection]
    ...

    Base your analysis solely on the provided information. Do not invent details. If no connections are found, state "No connections found" after # Connections: and explain why. Follow the format strictly.

    Let's think step by step.
    """

    connection_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "Dossiers:\n{all_docs}\n")
        ],
    )
    messages = connection_prompt_template.format_messages(all_docs=all_docs)
    # number_of_tokens = llm['openAI']['gpt4'].get_num_tokens_from_messages(messages)
    # print(number_of_tokens)

    #calling the prompt
    resp = llm['openAI']['gpt4'].invoke(messages).content.strip()
    # number_of_tokens = llm_gpt35['openAI'].get_num_tokens_from_messages(resp)
    return resp


def gpt_connection_cot_expanded(llm, all_docs):
    system_message = """You are an FBI agent, working with dossiers of multiple persons. Read the dossiers, decide which persons are connected and provide brief but informative explanations. Only list the pairs with direct connections. Follow this format strictly:

    # Connections: number of connections found

    Pair 1. Person A and Person B: [Brief explanation of their connection]
    Pair 2. Person A and Person C: [Brief explanation of their connection]
    Pair 3. Person D and Person E: [Brief explanation of their connection]
    ...

    Base your analysis solely on the provided information. Do not invent details. If no connections are found, state "No connections found" after # Connections: and explain why. Follow the format strictly.

    Let's think step by step:
    1. List all the people mentioned in the dossiers
    2. For each person, note their key attributes (locations, organizations, events, dates, etc.)
    3. Systematically compare each person with every other person to identify potential connections
    4. For each potential connection, evaluate the evidence that supports it.
    5. Output results in the specified format.
    """

    connection_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "Dossiers:\n{all_docs}\n")
        ],
    )
    messages = connection_prompt_template.format_messages(all_docs=all_docs)
    # number_of_tokens = llm['openAI']['gpt4'].get_num_tokens_from_messages(messages)
    # print(number_of_tokens)

    #calling the prompt
    resp = llm['openAI']['gpt4'].invoke(messages).content.strip()
    # number_of_tokens = llm_gpt35['openAI'].get_num_tokens_from_messages(resp)
    return resp