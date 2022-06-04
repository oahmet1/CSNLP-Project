import json
import penman
from penman import surface
import dgl
import torch
import pickle
from rdflib import Graph, Literal
import sys
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import matplotlib.pyplot as plt


def parse_amr(obj, remove_isolate_nodes=True):

    # TODO this is still from MRP and needs to be changed
    print(obj[0])
    print(obj[1])

    alignments = obj[1]
    penman_graph = penman.decode(obj[0])
    amr_nodes = set(penman_graph.variables())
    edge_types = set()

    print('vars', penman_graph.variables())
    print('instances', penman_graph.instances())

    for edge in penman_graph.edges():
        source, role, target = edge.source, edge.role, edge.target
        amr_nodes.add(source)
        amr_nodes.add(target)
        # v stands for vertex - not sure if we need multiple vertices in the future.
        edge_types.add(('v', role, 'v'))

    for attribute in penman_graph.attributes():
        source, role, target = attribute.source, attribute.role, attribute.target
        # TODO : add remove attributes or find embeddings for them
        amr_nodes.add(source)
        amr_nodes.add(target)
        edge_types.add(('v', role, 'v'))

    print("amr_nodes : ", amr_nodes)
    print("edge types :  ", edge_types)

    graph = Graph()

    if len(amr_nodes) == 0 or len(edge_types) == 0:
        return graph, []

    connected_nodes = set()
    for edge in penman_graph.edges():
        source, role, target = edge
        if source != -1 and target != -1:
            connected_nodes.add(source)
            connected_nodes.add(target)

    for attribute in penman_graph.attributes():
        source, role, target = attribute
        if source != -1 and target != -1:
            connected_nodes.add(source)
            connected_nodes.add(target)

    graph_nodes = []
    graph_metadata = []

    node_to_index = dict(list(map(lambda j: (j[1], j[0]), enumerate(list(amr_nodes)))))
    index_to_new_index = {}


    for node in amr_nodes:

        node_id = node_to_index[node]
        # if we want to remove it skip it, meaning we dont add it to the graph we are creating
        if remove_isolate_nodes and node not in connected_nodes:
            continue

        node_idx = len(graph_nodes)
        assert node_id not in index_to_new_index
        index_to_new_index[node_id] = node_idx

        graph_nodes.append(Literal(node_idx))

    print(index_to_new_index)
    print(node_to_index)
    # extract all the anchors

    graph_to_token_alignments = {}

    for key, value in surface.alignments(penman_graph).items():
        node = key[0]
        node_id = index_to_new_index[node_to_index[node]]
        graph_to_token_alignments[node_id] = value.indices

    print(index_to_new_index)
    print(node_to_index)

    for id, node in enumerate(graph_nodes):
        metadata = {}
        if id in graph_to_token_alignments:
            corresponding_token = graph_to_token_alignments[id][0]
            # TODO change here if you want to support multialignment.
            anchors = alignments[corresponding_token]
            print(anchors)
            metadata['anchors'] = anchors

        graph_metadata.append(metadata)


    print("graphToTokenAlignment : ", graph_to_token_alignments)

    role2literal = {}
    for edge in penman_graph.edges():
        source, role, target = edge
        if role not in role2literal:
            role2literal[role] = Literal(role)

        graph.add((graph_nodes[index_to_new_index[node_to_index[source]]], role2literal[role], graph_nodes[index_to_new_index[node_to_index[target]]]))

    for attribute in penman_graph.attributes():
        source, role, target = attribute
        if role not in role2literal:
            role2literal[role] = Literal(role)

        graph.add((graph_nodes[index_to_new_index[node_to_index[source]]], role2literal[role], graph_nodes[index_to_new_index[node_to_index[target]]]))



    if len(graph) == 0:
        print('WARNING: EMPTY GRAPH')

    print(graph.serialize(format="nt").decode('ascii'))
    print()
    print(graph_metadata)
    print('\n\n\n')


    return graph, graph_metadata


def process(file_name):
    with open(file_name) as f:
        json_string = json.load(f)

    graph_list = []
    metadata_list = []
    for sample in json_string:
        graph, metadata = parse_amr(sample)
        graph_list.append(graph)
        metadata_list.append(metadata)


    return graph_list, metadata_list

'''
        samplenumber = samplenumber + 1
        print()
        # first we decode the penman string
        penman_graph = penman.decode(sample[0])

        token_to_char_alignments = sample[1]
        graph_to_token_alignments = surface.alignments(penman_graph)
        print(penman_graph.metadata["snt"])
        print("edges :   ", penman_graph.edges())
        print(sample[0])

        nodes = set(penman_graph.variables())
        edge_types = set()

        print('vars', penman_graph.variables())
        print('instances', penman_graph.instances())


        if len(penman_graph.edges()) != 0:
            for edge in penman_graph.edges():
                source, role, target = edge.source, edge.role, edge.target
                #nodes.add(source)
                #nodes.add(target)
                # v stands for vertex - not sure if we need multiple vertices in the future.
                edge_types.add(('v', role, 'v'))

        if len(penman_graph.attributes()) != 0:
            for attribute in penman_graph.attributes():
                source, role, target = attribute.source, attribute.role, attribute.target
                #TODO : add remove attributes or find embeddings for them
                nodes.add(source)
                nodes.add(target)
                edge_types.add(('v', role, 'v'))

        print("nodes : " , nodes)
        print("edge types :  " , edge_types)

        # %%
        edge_collection = {}


        for edge_type in edge_types:
        # v1, t, v2  = edge_type
            edge_collection[edge_type] = []

        node_to_index = dict(list(map(lambda j: (j[1], j[0]), enumerate(list(nodes)))))

        #print(node_to_index)

        if len(penman_graph.edges()) != 0:
            for edge in penman_graph.edges():
                source, role, target = edge.source, edge.role, edge.target
                print("the role : ", role)
                edge_collection[('v', role, 'v')].append((node_to_index[source], node_to_index[target]))

        if len(penman_graph.attributes()) != 0:
            for attribute in penman_graph.attributes():
                source, role, target = attribute.source, attribute.role, attribute.target
                edge_collection[('v', role, 'v')].append((node_to_index[source], node_to_index[target]))

        dgl_graph = {}
        u = []
        v = []

        for edge_type in edge_collection:
            print(f'edge type : {edge_type}')
            print(f'edge collection : {edge_collection}')

            print(*edge_collection[edge_type])

            u, v = list(zip(*edge_collection[edge_type]))
            u, v = list(u), list(v)
            dgl_graph[edge_type] = torch.tensor(u), torch.tensor(v)

        graph_to_token_alignments = {}

        node_to_index = dict(list(map(lambda j: (j[1], j[0]), enumerate(list(nodes)))))

        for key, value in surface.alignments(penman_graph).items():
            node = key[0]
            node_id = node_to_index[node]
            graph_to_token_alignments[node_id] = value.indices

        print(node_to_index)
        print("graphToTokenAlignment : " ,graph_to_token_alignments)


        # we give back: the DGL graph, the node to token alignments and the token to character alignments
        # TODO node to token is currently the wrong thing
'''



if __name__ == '__main__':
    input_file, graph_output_file, metadata_output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    #input_file = 'amr_data_cola_train.json'
    all_graphs, all_metadata = process(input_file)

    pickle.dump(all_graphs, open(graph_output_file, 'wb'))
    pickle.dump(all_metadata, open(metadata_output_file, 'wb'))

