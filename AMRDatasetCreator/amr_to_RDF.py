import json
import penman
import dgl
from penman import surface
import torch
import pickle

from penman.graph import Attribute, Instance
from penman.models.noop import NoOpModel
from rdflib import Graph, Literal
import sys
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import matplotlib.pyplot as plt

DEBUG = False


def parse_amr(obj, remove_isolate_nodes=True):

    if DEBUG:
        print(obj[0])
        print(obj[1])
        print(penman.encode(obj[0]))

    penman_graph = obj[0] # penman.decode(obj[0], model=NoOpModel())
    token_range = obj[1]

    penman_amr_nodes_set = set(penman_graph.variables())
    penman_edge_types = set()

    if DEBUG:
        print('vars', penman_graph.variables())
        print('instances', penman_graph.instances())

        print('edges: ', penman_graph.edges())
        print('attributes: ', penman_graph.attributes())

    for edge in penman_graph.edges():
        source, role, target = edge.source, edge.role, edge.target
        # v stands for vertex - not sure if we need multiple vertices in the future.
        penman_edge_types.add(('v', role, 'v'))

    # TODO not needed in future?
    for attribute in penman_graph.attributes():
        source, role, target = attribute.source, attribute.role, attribute.target
        # TODO : add remove attributes or find embeddings for them
        penman_amr_nodes_set.add(source)
        penman_amr_nodes_set.add(target)
        penman_edge_types.add(('v', role, 'v'))

    if DEBUG:
        print("amr_nodes : ", penman_amr_nodes_set)
        print("edge types :  ", penman_edge_types)

    rdf_graph = Graph()

    if len(penman_amr_nodes_set) == 0 or len(penman_edge_types) == 0:
        return rdf_graph, []

    connected_amr_nodes = set()
    for edge in penman_graph.edges():
        source, role, target = edge
        if source != -1 and target != -1:
            connected_amr_nodes.add(source)
            connected_amr_nodes.add(target)

    for attribute in penman_graph.attributes():
        source, role, target = attribute
        if source != -1 and target != -1:
            connected_amr_nodes.add(source)
            connected_amr_nodes.add(target)

    graph_nodes = []
    graph_metadata = []

    node_to_index = dict(list(map(lambda j: (j[1], j[0]), enumerate(list(penman_amr_nodes_set)))))
    index_to_new_index = {}

    for node in penman_amr_nodes_set:

        node_id = node_to_index[node]
        # if we want to remove it skip it, meaning we dont add it to the graph we are creating
        if remove_isolate_nodes and node not in connected_amr_nodes:
            continue

        node_idx = len(graph_nodes)
        assert node_id not in index_to_new_index
        index_to_new_index[node_id] = node_idx

        graph_nodes.append(Literal(node_idx))

    if DEBUG:
        print(index_to_new_index)
        print(node_to_index)
    # extract all the anchors

    graph_to_token_alignments = {}

    for key, value in surface.alignments(penman_graph).items():
        node = key[0]
        node_id = index_to_new_index[node_to_index[node]]
        if DEBUG:
            print('align:', key, value, node_id)
        if node_id not in graph_to_token_alignments:
            graph_to_token_alignments[node_id] = []

        for v in value.indices:
            graph_to_token_alignments[node_id].append(v)

    if DEBUG:
        print(index_to_new_index)
        print(node_to_index)
        print('tokenalignments are: ', graph_to_token_alignments)

    # print(f"graph to token alignments:   {graph_to_token_alignments}")
    for id, node in enumerate(graph_nodes):
        metadata = {}
        if id in graph_to_token_alignments:
            corresponding_tokens = graph_to_token_alignments[id]
            anchors = []
            for token in corresponding_tokens:
                anchors.append(token_range[token])
            if DEBUG:
                print(anchors)
            metadata['anchors'] = anchors
        else:
            # we are at a node that is not aligned
            # code source: https://stackoverflow.com/questions/2568673/inverse-dictionary-lookup-in-python
            prev_id = next(key for key, value in index_to_new_index.items() if value == id)
            #print(f"prev_id:   {prev_id}")
            penman_id = next(key for key, value in node_to_index.items() if value == prev_id)
            # first check if it is an instance
            instance_list = list(filter(lambda x: x.source == penman_id, penman_graph.instances()))
            attribute_list = list(filter(lambda x: x.target == penman_id, penman_graph.attributes()))

            if len(instance_list) > 0:
                target = instance_list[0].target
            elif len(attribute_list) > 0:
                target = attribute_list[0].target
            else:
                target = None

            # target handling

            if target == "-":
                transformed_target = 'not'
            elif target[0] == '\"' and target[-1] == '\"':
                transformed_target = target.strip('"')
            elif target.isnumeric():
                transformed_target = target
            else:
                transformed_target = target.strip('-01234567890')
                # transformed_target=None

            # print(transformed_target)
            # relevant_penman_triple = list(filter(lambda x: x[0] == penman_id, penman_graph.triples))
            metadata['amr_unaligned'] = transformed_target

        graph_metadata.append(metadata)

    if DEBUG:
        print("graphToTokenAlignment : ", graph_to_token_alignments)

    role2literal = {}
    for edge in penman_graph.edges():
        source, role, target = edge
        if role not in role2literal:
            role2literal[role] = Literal(role)

        rdf_graph.add((graph_nodes[index_to_new_index[node_to_index[source]]], role2literal[role], graph_nodes[index_to_new_index[node_to_index[target]]]))

    for attribute in penman_graph.attributes():
        source, role, target = attribute
        if role not in role2literal:
            role2literal[role] = Literal(role)

        rdf_graph.add((graph_nodes[index_to_new_index[node_to_index[source]]], role2literal[role], graph_nodes[index_to_new_index[node_to_index[target]]]))

    if len(rdf_graph) == 0:
        print('WARNING: EMPTY GRAPH')

    if DEBUG:
        print(rdf_graph.serialize(format="nt").decode('ascii'))
        print()
        print(graph_metadata)
        print('\n\n\n')

    return rdf_graph, graph_metadata


def process(file_name):
    with open(file_name, 'rb') as f:
        json_string = pickle.load(f)

    graph_list = []
    metadata_list = []
    for sample in json_string:
        graph, metadata = parse_amr(sample)
        graph_list.append(graph)
        metadata_list.append(metadata)


    return graph_list, metadata_list




if __name__ == '__main__':
    input_file, graph_output_file, metadata_output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    all_graphs, all_metadata = process(input_file)
    # print(all_metadata)
    pickle.dump(all_graphs, open(graph_output_file, 'wb'))
    pickle.dump(all_metadata, open(metadata_output_file, 'wb'))

