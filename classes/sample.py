import os
import pandas as pd
import json
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
from collections import defaultdict


# Take the graph input and Sample some entities from there based on the provided parameters
# Sample some entities from the graph
import random

def load_documents_by_filename(input_directory, dataset, filenames):
    result = []
    all_docs = load_documents(input_directory, dataset)
    return [x.content for x in all_docs if x['filename'] in filenames]

def get_dot_by_id(list_of_dots, dot_id):
    if type(list_of_dots[0]) == InfoDot:
        for dot in list_of_dots:
            if dot.id == dot_id:
                return dot
    else:
        for dot in list_of_dots:
            if dot['id'] == dot_id:
                return dot
    return None


def load_person_data_graph(person_file, adjacency_mat_file): 
    assert os.path.exists(person_file), f"Person file does not exist: {person_file}"
    assert os.path.exists(adjacency_mat_file), f"Adjacency matrix file does not exist: {adjacency_mat_file}"

    # load the person data
    with open(person_file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    # print(len(data), data[0])

    # read the adjacency dataframe, switch the adjacency matrix to manual one after completion
    adjacency_mat = pd.read_csv(adjacency_mat_file, index_col=0, encoding='cp1252')
    # print(adjacency_mat.head(2))

    # using eval make a list of InfoDot
    persons = [InfoDot(**x) for x in data]
    print(persons[0])

    # initialize a new graph
    connection_graph = nx.Graph()
    connection_graph.add_nodes_from(persons)
    # print(connection_graph)

    # after making the adjacency mats, load the edges to the graph
    for index, row in adjacency_mat.iterrows():
        for column, value in row.items():
            # value = int(value)
            if not pd.isna(value):
                # print(value)
                id_1 = eval(index)['id']
                id_2 = eval(column)['id']
                print(id_1, id_2)
                node_1 = get_dot_by_id(list(connection_graph.nodes), id_1)
                node_2 = get_dot_by_id(list(connection_graph.nodes), id_2)
                connection_graph.add_edge(node_1, node_2, story=value)
                # print(node_1, type(node_2))

    # Make a connection proxy graph with only id numbers as node so that sampling algorithms run better
    connection_proxy_graph = generate_proxy_graph(connection_graph)
    print("connections graph and proxy graph created: ", connection_proxy_graph)
    return connection_proxy_graph, connection_graph, persons


# take the connection graph and produce a proxy graph with only the ids of the instances
def generate_proxy_graph(graph):
    proxy_graph = nx.Graph()
    for node in graph.nodes():
        proxy_graph.add_node(node.id)
    for edge in graph.edges():
        proxy_graph.add_edge(edge[0].id, edge[1].id)
    return proxy_graph

def flatten_set(xss):
    xss = [list(x) for x in xss]
    return xss, [x for xs in xss for x in xs]

def flatten_list(xss):
    flattened_list = [x for xs in xss for x in xs]
    return flattened_list, list(set(flattened_list))

def sample_test(persons, connection_proxy_graph, **params):
    # print(params)
    random.seed(params['seed'])

    if params['type'] == 'edge':
        # Read the params['connection_dict']
        connected_nodes = params['connection_dict']['connected_nodes']
        disconnected_nodes = params['connection_dict']['disconnected_nodes'][-1]
    
        sample_list = []

        for i in range(0, params["num_total_samples"]):
            # make a list of nodes from these components
            sampled_connected_nodes = random.sample(connected_nodes, params['connections_per_sample'])

            # newly added
            non_sampled_connected_nodes = [x for x in connected_nodes if x not in sampled_connected_nodes]
            # take one from each tuple in non_sampled_connected_nodes by sampling
            non_sampled_connected_nodes = [random.choice(x) for x in non_sampled_connected_nodes]
            # print("non_sampled_connected_nodes", non_sampled_connected_nodes)

            # updated
            sampled_disconnected_nodes = random.sample(disconnected_nodes+non_sampled_connected_nodes, params['total_entities'] - params['connections_per_sample']*2)

            # disperse these sets into a list
            sampled_connected_nodes, sampled_connected_nodes_flatten = flatten_set(sampled_connected_nodes)
            # sampled_connected_nodes = [get_dot_by_id(persons, x) for x in sampled_connected_nodes]
            # print("sample connected nodes", sampled_connected_nodes)
            # print("sample disconnected nodes", sampled_disconnected_nodes)
            
            sampled_nodes = sampled_connected_nodes_flatten + sampled_disconnected_nodes
            # shuffle the list with provided seed
            random.shuffle(sampled_nodes)

            all_docs = ""
            for item in sampled_nodes:
                person = get_dot_by_id(persons, item)
                all_docs += f"{person.name}: {person.info}\n\n"
                # print(item) 
            # print(all_docs)
            
            test_dict = {
                "id": f"test_{i}",
                # "params": params,
                "all_docs": all_docs,
                # "prompt": messages,
                "sampled_nodes_together_with_order": sampled_nodes,
                "sampled_connected_nodes": sampled_connected_nodes,
                "sampled_disconnected_nodes": sampled_disconnected_nodes,
            }
            sample_list.append(test_dict)


    elif params['type'] == 'degree' or params['type'] == 'clique':
        size_range = range(params["entities_per_connection_min"], params["entities_per_connection_max"] + 1)
        size_range = [x for x in size_range if x in params['connection_dict'].keys()]

        sample_list = []
        num_sample_per_degree = params["num_total_samples"]//len(size_range)

        test_number = 0
        for degree in size_range:
            connected_nodes = params['connection_dict'][degree]['connected_nodes']
            disconnected_nodes = params['connection_dict'][degree]['disconnected_nodes'][-1]
            num_connected_entities_for_this_degree = degree

            # For this, we will divide the num of total sample into degree_min and degree_max and get a sample from each
            for i in range(num_sample_per_degree):

                # make a list of nodes from these components
                sampled_connected_nodes = random.sample(connected_nodes, params['connections_per_sample'])
                sampled_disconnected_nodes = random.sample(disconnected_nodes, params['total_entities'] - params['connections_per_sample']*num_connected_entities_for_this_degree)
                # disperse these sets into a list
                sampled_connected_nodes, sampled_connected_nodes_flatten = flatten_set(sampled_connected_nodes)
                # sampled_connected_nodes = [get_dot_by_id(persons, x) for x in sampled_connected_nodes]
                # print("sample connected nodes", sampled_connected_nodes)
                # print("sample disconnected nodes", sampled_disconnected_nodes)
                
                sampled_nodes = sampled_connected_nodes_flatten + sampled_disconnected_nodes
                # shuffle the list with provided seed
                random.shuffle(sampled_nodes)

                all_docs = ""
                for item in sampled_nodes:
                    person = get_dot_by_id(persons, item)
                    all_docs += f"{person.name}: {person.info}\n\n"
                    # print(item) 
                # print(all_docs)

                # make the adjacency matrix from the original graph with only sampled nodes (both connected and disconnected nodes)
                subgraph_with_selected_nodes = connection_proxy_graph.subgraph(sampled_nodes)
                adjacency_matrix = nx.adjacency_matrix(subgraph_with_selected_nodes, nodelist=sorted(list(subgraph_with_selected_nodes.nodes()))).todense()
                # print("nodes", subgraph_with_selected_nodes.nodes)
                # print("sorted nodes", sorted(list(subgraph_with_selected_nodes.nodes())))
                # print("adjacency matrix", adjacency_matrix)
                # print("sampled nodes", sampled_nodes)
                # print("to list", adjacency_matrix.tolist())
                # print()

                test_dict = {
                    "id": f"test_{test_number}",
                    # "params": params,
                    "all_docs": all_docs,
                    # "prompt": messages,
                    "sampled_nodes_together_with_order": sampled_nodes,
                    "sampled_connected_nodes": sampled_connected_nodes,
                    "sampled_disconnected_nodes": sampled_disconnected_nodes,
                    "degree_clique_size": degree,
                    "type": params['type'],
                    "adjacency_matrix": adjacency_matrix.tolist(),
                }
                sample_list.append(test_dict)
                test_number+=1

    return sample_list


def disperse_connected_nodes(sampled_list, start, end, llm, persons, type="edge"):
    for item in sampled_list:
        connected_nodes = item['sampled_connected_nodes']  # List of tuples
        disconnected_nodes = item['sampled_disconnected_nodes']  # List of integers

        num_connected_groups = len(connected_nodes)  # Number of tuples
        num_elements_per_tuple = len(connected_nodes[0])  # Number of elements per tuple
        total_length = len(disconnected_nodes)

        # Convert start and end from percentage to absolute index positions
        start_idx = int(np.floor(start * total_length)) if isinstance(start, float) else start
        end_idx = int(np.floor(end * total_length)) if isinstance(end, float) else end

        print("start_idx", start_idx, "end_idx", end_idx)

        # Ensure valid range
        if start_idx < 0 or end_idx > total_length or start_idx >= end_idx:
            raise ValueError("Start and end indices are out of bounds or invalid")
        
        # make set of first element of each tuple, second element from each tuple        # and so on
        insert_list = []
        for i in range(num_elements_per_tuple):
            # get the elements from each tuple
            insert_set = [connected_nodes[j][i] for j in range(num_connected_groups)]
            insert_list.append(insert_set)

        print("insert_list", insert_list)

        reverse_insert_list = []
        for i in range(num_elements_per_tuple):
            # get the elements from each tuple
            reverse_insert_set = [connected_nodes[j][-(i+1)] for j in range(num_connected_groups)]
            reverse_insert_list.append(reverse_insert_set)

        print("reverse_insert_list", reverse_insert_list)


        # split the disconnected nodes into the num_element_per_tuple parts
        splitting_indices = np.linspace(start_idx, end_idx, num_elements_per_tuple).astype(int)
        print("splitting indices", splitting_indices)
        
        new_list = []
        new_list_reverse = []
        for i, ind in enumerate(splitting_indices):
            if i == 0:
                new_list.extend(disconnected_nodes[:ind])
                new_list_reverse.extend(disconnected_nodes[:ind])
            else:
                new_list.extend(disconnected_nodes[splitting_indices[i-1]:ind])
                new_list_reverse.extend(disconnected_nodes[splitting_indices[i-1]:ind])
            # add the elements from the tuples
            new_list.extend(insert_list[i])
            new_list_reverse.extend(reverse_insert_list[i])

            # if i is equal to the len of splitting index, add the last part of the disconnected nodes
            if i == len(splitting_indices) - 1:
                new_list.extend(disconnected_nodes[ind:])
                new_list_reverse.extend(disconnected_nodes[ind:])

        # Update item with the new list of nodes
        item["sampled_nodes_together_with_order"] = new_list
        item["sampled_nodes_together_with_order_reverse"] = new_list_reverse

        # calculate the number of items between each element of the tuple, it's same for all, so just take the first one
        distance_between_first_last = new_list.index(connected_nodes[0][-1]) - new_list.index(connected_nodes[0][0])
        avg_distance_between = distance_between_first_last
        # going with the average distance between the first and last element of the tuple
        # avg_distance_between = distance_between_first_last / (num_elements_per_tuple - 1)
        item["num_items_between"] = avg_distance_between

        # calculate the token count for each sampled list
        avg_token_count, token_count = get_tokens_between(item, llm, persons)
        item['avg_token_count'] = avg_token_count
        item['token_count'] = token_count

        # update all_doc with new item["sampled_nodes_together_with_order"] and item["sampled_nodes_together_with_order_reverse"], very important, so two new. all_docs and all_docs_reverse

        all_docs = ""
        for node in item["sampled_nodes_together_with_order"]:
            person = get_dot_by_id(persons, node)
            all_docs += f"{person.name}: {person.info}\n\n"
        item["all_docs"] = all_docs
        all_docs_reverse = ""
        for node in item["sampled_nodes_together_with_order_reverse"]:
            person = get_dot_by_id(persons, node)
            all_docs_reverse += f"{person.name}: {person.info}\n\n"
        item["all_docs_reverse"] = all_docs_reverse

    return sampled_list



def get_tokens_between(sampled_list_item, llm, persons):
    connected_nodes = sampled_list_item['sampled_connected_nodes']  # List of tuples
    new_list = sampled_list_item['sampled_nodes_together_with_order']  # List of integers
    # get actual token count between the tuple elements:
    doc_list = []
    for i, tuples in enumerate(connected_nodes):
        elements_between_first_last = new_list[new_list.index(tuples[0]):new_list.index(tuples[-1])]
        doc_for_this_gap = ""
        for element in elements_between_first_last:
            person = get_dot_by_id(persons, element)
            doc_for_this_gap += f"{person.name}: {person.info}\n\n"
        
        doc_list.append(doc_for_this_gap)
    
    # calculate the token count for each doc
    token_count = []
    for doc in doc_list:
        # print("doc", doc)
        
        token_count.append(llm['openAI']['gpt4'].get_num_tokens(doc))

    return np.average(token_count), token_count


##########################
# Util necessary for the graph operations
import networkx as nx
from copy import deepcopy
from networkx.algorithms import approximation

def make_disconnected(graph, method="vertex_cover"):
    """
    methods can be 'traditional' or 'heuristic' or 'vertex cover'
    """
    # Create a copy of the graph to avoid modifying the original
    graph_copy = graph.copy()

    if method == 'traditional':
        # raise NotImplementedError("Traditional method is not implemented yet")
        # Identify all nodes that are connected by at least one edge
        nodes_with_edges = {node for edge in graph_copy.edges() for node in edge}
        # Remove all edges
        graph_copy.remove_edges_from(list(graph_copy.edges()))
        # Remove nodes that had at least one edge before removal
        graph_copy.remove_nodes_from(nodes_with_edges)
    elif method == 'heuristic':
        # remove the nodes one by one starting with highest degree and check if all nodes are isolate
        raise NotImplementedError("Heuristic method is not implemented yet")
    elif method == 'vertex_cover':
        # print("Using vertex cover method")
        vertex_cover = approximation.min_weighted_vertex_cover(graph_copy)
        graph_copy.remove_nodes_from(vertex_cover)
        # assert the remaining nodes in graph_copy are isolates
        assert all([x[1] == 0 for x in list(graph_copy.degree)])
    else:
        raise ValueError(f"Method {method} is not supported, Mention one of 'traditional', 'heuristic', 'vertex cover'")
    
    return graph_copy


##########################
# Edge based Sampling below

import networkx as nx
from copy import deepcopy
from tqdm import tqdm

def calculate_edge_degrees(graph):
    edge_degrees = {}
    for edge in graph.edges():
        node_a, node_b = edge
        degree_a = graph.degree(node_a)
        degree_b = graph.degree(node_b)
        combined_degree = degree_a + degree_b
        edge_degrees[edge] = combined_degree
    return edge_degrees

def run_heuristic_algorithm(graph):
    connected_nodes = []
    disconnected_nodes = []
    
    # Calculate combined degree count for each edge
    edge_degrees = calculate_edge_degrees(graph)
    
    # Make a deepcopy of the graph so the original is not modified
    graph_copy = deepcopy(graph)
    
    # Initial step: process the edge with the smallest combined degree
    while len(graph_copy.edges) > 0:
        # Find the edge with the minimum combined degree
        edge = min(edge_degrees, key=edge_degrees.get)
        edge_set = set(edge)
        connected_nodes.append(list(edge_set))
        
        # Remove the nodes of the edge and their neighbors
        nodes_to_remove = set(graph_copy.neighbors(edge[0])) | set(graph_copy.neighbors(edge[1])) | edge_set
        graph_copy.remove_nodes_from(nodes_to_remove)
        
        remaining_nodes = set(graph_copy.nodes())
        if remaining_nodes:
            disconnected_nodes.append(list(remaining_nodes))
        
        # Recalculate the combined degree counts after removing nodes
        edge_degrees = calculate_edge_degrees(graph_copy)

    return {'connected_nodes': connected_nodes, 'disconnected_nodes': disconnected_nodes}


#####################
# Common utils for Degree and clique based samplings
def calculate_degree_sum(graph, group_of_nodes):
    degree_dict = {}
    for nodes in group_of_nodes:
        degree_dict[tuple(nodes)] = sum([x[1] for x in list(graph.degree(nodes))])
    return degree_dict

def calculate_degree_with_neighbor(graph, nodes):
    degree_dict = {}
    for node in nodes:
        neighbors_with_node = list(graph.neighbors(node)) + [node]
        degree_dict[node] = sum([x[1] for x in list(graph.degree(neighbors_with_node))])
    return degree_dict


#####################
# Degree based Sampling below
import networkx as nx
# import random
from collections import defaultdict

# TODO: add minimum degree based choice instead of random, using "calculate_degree_sum"
def sample_and_classify(orig_graph, degree, seed):
    connected_nodes = []
    disconnected_nodes = []
    # random.seed(seed)
    graph = deepcopy(orig_graph)

    # Filter nodes to only those with the specified degree
    eligible_nodes = [node for node in graph.nodes if graph.degree[node] == degree]

    while len(eligible_nodes) > 0:
        eligible_nodes_degree_sum_dict = calculate_degree_with_neighbor(graph, eligible_nodes)
        sampled_node = min(eligible_nodes_degree_sum_dict, key=eligible_nodes_degree_sum_dict.get)
        
        # Add the node and its neighbors to connected_nodes
        neighbors = list(graph.neighbors(sampled_node))
        node_set_to_add = [sampled_node] + neighbors
        connected_nodes.append(node_set_to_add)

        assert sampled_node == node_set_to_add[0]; "the first node needs to be the center node"
        
        # calculate all neighboring nodes to node_set_to_add and Remove these nodes from the graph
        # all neighbors combined with the node_set_to_add
        # full_node_set = node_set_to_add + [node for node in neighbors]
        # print()
        _, node_set_to_remove = flatten_list([graph.neighbors(node) for node in node_set_to_add]+[node_set_to_add])
        # print("node_set_to_add", node_set_to_add, "node_set_to_remove", node_set_to_remove)
        graph.remove_nodes_from(node_set_to_remove)
        
        # Update the list of eligible nodes
        eligible_nodes = [node for node in graph.nodes if graph.degree[node] == degree]

        # call deconnect function
        disconnected_graph = deepcopy(graph)
        disconnected_graph = make_disconnected(disconnected_graph,)
        remaining_nodes = list(set(disconnected_graph.nodes))
        
        # Add all remaining nodes to disconnected_nodes (those that were not connected)
        # They don't need to be the eligible nodes, as they may have a different degree
        disconnected_nodes.append(remaining_nodes)
    
    # Test to make sure properties are maintained
    disconnected_subgraph = orig_graph.subgraph(disconnected_nodes[-1])
    assert all([x[1] == 0 for x in list(disconnected_subgraph.degree)]), "The disconnected nodes are not truly disconnected"

    # making sure connected+disconnected nodes subgraphs also maintain properties
    all_connected_nodes = [node for nodes in connected_nodes for node in nodes]
    full_subgraph = orig_graph.subgraph(all_connected_nodes+disconnected_nodes[-1])
    # print("degree: ",degree, "full subgraph nodes list: ", full_subgraph.nodes, "all connected nodes: ", all_connected_nodes, "disconnected nodes: ", disconnected_nodes[-1])
    assert all([x[1] == 0 for x in list(full_subgraph.degree(disconnected_nodes[-1]))]), "The disconnected nodes are not truly disconnected in full subgraph"
    assert all([x[1] > 0 for x in list(full_subgraph.degree(all_connected_nodes))]), "The connected nodes are not truly connected in full subgraph"

    return connected_nodes, disconnected_nodes

def organize_nodes_by_degree(graph):
    degree_dict = defaultdict(list)
    for node, degree in graph.degree():
        degree_dict[degree].append(node)
    return degree_dict

def run_algorithm_for_each_degree(graph, seed=42):
    degree_dict = organize_nodes_by_degree(graph)
    results_dict = {}

    for degree in degree_dict.keys():
        if degree==0:
            continue
        # Run the sample and classify algorithm with the whole graph and the specific degree
        connected, disconnected = sample_and_classify(graph.copy(), degree, seed=seed)
        
        # Store the results in the dictionary
        # save +1 in the degree key so that it follows the size of the connections
        results_dict[degree+1] = {
            "connected_nodes": connected,
            "disconnected_nodes": disconnected
        }

    return results_dict


#############################
# Clique based Sampling below
import networkx as nx
import random
from collections import defaultdict

# Clique based Sampling below
# Find the cliques on the graph

def make_clique_dict(graph):
    cliques = list(nx.find_cliques(graph))
    # classify the cliques based on the number of nodes
    clique_dict = defaultdict(list)
    for clique in cliques:
        clique_dict[len(clique)].append(clique)
    
    return clique_dict

def sample_and_classify_clique(orig_graph, clique_dict, clique_size, seed):
    connected_nodes = []
    disconnected_nodes = []
    # random.seed(seed)
    graph = deepcopy(orig_graph)

    # Filter nodes to only those with the specified degree
    eligible_nodes = clique_dict[clique_size]
    # print("eligible_nodes", eligible_nodes)

    while len(eligible_nodes) > 0:
        eligible_nodes_degree_sum_dict = calculate_degree_sum(graph, eligible_nodes)
        node_set_to_add = min(eligible_nodes_degree_sum_dict, key=eligible_nodes_degree_sum_dict.get)
        # print("node_set_to_remove", node_set_to_remove, type(node_set_to_remove))
        connected_nodes.append(node_set_to_add)

        # print("graph before:", graph.nodes)
        # Remove these nodes from the graph, and their neighbors
        _, node_set_to_remove = flatten_list([graph.neighbors(node) for node in node_set_to_add]+[node_set_to_add])
        graph.remove_nodes_from(node_set_to_remove)
        # print("graph after:", graph.nodes)
        
        # Update the list of eligible nodes
        new_eligible_nodes = [node_set for node_set in eligible_nodes if all([node in graph.nodes for node in node_set])]
        assert len(new_eligible_nodes) < len(eligible_nodes), "No nodes were removed from the graph"
        eligible_nodes = new_eligible_nodes

        # Call the deconnect function
        disconnected_graph = deepcopy(graph)
        disconnected_graph = make_disconnected(disconnected_graph,)
        remaining_nodes = list(set(disconnected_graph.nodes))
        
        # Add all remaining nodes to disconnected_nodes (those that were not connected)
        # They don't need to be the eligible nodes, as they may have a different degree
        disconnected_nodes.append(remaining_nodes)

    # Test to make sure properties are maintained
    disconnected_subgraph = orig_graph.subgraph(disconnected_nodes[-1])
    assert all([x[1] == 0 for x in list(disconnected_subgraph.degree)]), "The disconnected nodes are not truly disconnected"

    # making sure connected+disconnected nodes subgraphs also maintain properties
    all_connected_nodes = [node for nodes in connected_nodes for node in nodes]
    full_subgraph = orig_graph.subgraph(all_connected_nodes+disconnected_nodes[-1])
    # print("clique_size: ",clique_size, "full subgraph nodes list: ", full_subgraph.nodes, "all connected nodes: ", all_connected_nodes, "disconnected nodes: ", disconnected_nodes[-1])
    # print("degree list of full subgraph", list(full_subgraph.degree))
    assert all([x[1] == 0 for x in list(full_subgraph.degree(disconnected_nodes[-1]))]), "The disconnected nodes are not truly disconnected in full subgraph"
    assert all([x[1] > 0 for x in list(full_subgraph.degree(all_connected_nodes))]), "The connected nodes are not truly connected in full subgraph"

    return connected_nodes, disconnected_nodes

def run_algorithm_for_each_clique_size(graph, seed=42):
    clique_dict = make_clique_dict(graph)
    results_dict = {}

    for clique_size in clique_dict.keys():
        if clique_size<=1:
            continue
        # Run the sample and classify algorithm with the whole graph and the specific degree
        connected, disconnected = sample_and_classify_clique(graph.copy(), clique_dict, clique_size, seed=seed)
        
        # Store the results in the dictionary
        results_dict[clique_size] = {
            "connected_nodes": connected,
            "disconnected_nodes": disconnected
        }

    return results_dict

