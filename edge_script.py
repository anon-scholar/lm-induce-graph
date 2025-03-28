import os
import pandas as pd
import json
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import pickle
import networkx as nx
from classes.util import (InfoDot, load_full_model, load_quantized_model_qptq, load_model)
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import re
import numpy as np
import matplotlib.pyplot as plt
from classes.sample import *
from classes.util_new import *


from transformers import (
    GenerationConfig,
    pipeline,
)

from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    
)

if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


adjacency_mat_file = 'data/atlantic_storm/at_done.csv'
person_file = "data/atlantic_storm/persons.json"
connection_proxy_graph, connection_graph, persons = load_person_data_graph(person_file, adjacency_mat_file)

# TODO: Change this to a function
name_id_dict = {}
for person in persons:
  name_id_dict[person.get_dict().get('name')] = person.get_dict().get('id')



# Run the edge based heuristic algorithm
from classes.sample import *

connection_dict = run_heuristic_algorithm(connection_proxy_graph)

# Print the results
print("Connected Nodes Sets:")
print(type(connection_dict['connected_nodes']))
for cn in connection_dict['connected_nodes']:
    print(cn)
print("\nDisconnected Nodes Sets:")
for dn in connection_dict['disconnected_nodes']:
    print(dn)



llm = load_model(model_type="openAI", openai_temperature=0.1)

# saving the responses with sampling information and metrics and test with gpt4,
"""
EDGE type
"""
tot_entities = range(1, 100)
tot_connection_per_sample = range(1, 20)
num_total_samples = 10

full_results = []

for total_entities in tot_entities:
    for connections_per_sample in tot_connection_per_sample:
        print("total_entities: ", total_entities, "connections_per_sample: ", connections_per_sample)
        try:
            params = {
                "seed": 42,
                "total_entities": total_entities,
                "type": "edge",
                "connections_per_sample": connections_per_sample,
                "entities_per_connection_min": 2,
                "entities_per_connection_max": 2,
                "connection_length_max": 1,
                "connection_length_min": 1,
                "num_total_samples": num_total_samples,
                "connection_dict": connection_dict,
            }
            sampled_list = sample_test(persons, connection_graph, **params)
            dispersed_sampled_list = disperse_connected_nodes(sampled_list, 0.1, 1.0, llm, persons)
            print("worked\n")
        except Exception as e:
            print(f"For this param, an Error: {e} occured ")
            print("skipping this param\n")
            continue
        
        # iterate through the sampled_list and for each example, get response and metrics and append them
        metrics = []

        for id, sampled_list_item in enumerate(dispersed_sampled_list):
            try:
                resp = gpt_connection(llm, sampled_list_item.get('all_docs'))
                # print(resp)
                extracted = extract_connections_pair(resp)
                
                predicted_connections = get_id_from_name(extracted, name_id_dict)
                # print("predicted_connections:", predicted_connections)

                true_connections = [set(i) for i in sampled_list_item['sampled_connected_nodes']]

                true_adj_matrix = generate_adjacency_matrix(sorted(sampled_list_item['sampled_nodes_together_with_order']), true_connections)
                predicted_adj_matrix = generate_adjacency_matrix(sorted(sampled_list_item['sampled_nodes_together_with_order']), predicted_connections)

                score, full_metric = calculate_score(true_adj_matrix, predicted_adj_matrix)

                metrics.append({
                    "sampled_example": sampled_list_item,
                    "response": resp,
                    "predicted_connections": [list(x) for x in predicted_connections],
                    "score": score,
                    "full_metric": full_metric,
                    "avg_token_count": sampled_list_item.get('avg_token_count'),
                })
                # print("-"*20)
            except Exception as e:
                print(f"Error: {e} occured ")
                continue

        if len(metrics) > 0:
            # get an average of all the scores
            avg_score = sum([metric['score'] for metric in metrics])/len(metrics)
            avg_token_count_for_this_setting = sum([metric['avg_token_count'] for metric in metrics])/len(metrics)
                
            full_result_dictionary = {
                # "params": params,
                "total_entities": total_entities,
                "connections_per_sample": connections_per_sample,
                # "original_sampled_list": sampled_list,
                "metrics": metrics,
                "avg_score": avg_score,
                "avg_token_count_for_this_setting": avg_token_count_for_this_setting,

            }

            # save the full_result_dictionary to a jsonl file in append mode
            with open('output/full_results_edge_at.jsonl', 'a') as f:
                f.write(json.dumps(full_result_dictionary) + '\n')

            # append the full_result_dictionary to a list
            full_results.append(full_result_dictionary)


with open('output/full_results_edge_at.json', 'w') as f:
    json.dump(full_results, f, indent=4)