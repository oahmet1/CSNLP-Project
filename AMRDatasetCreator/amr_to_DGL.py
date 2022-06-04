import json
import penman
from penman import surface
import dgl
import torch
import pickle


def process(file_name):
    with open(file_name) as f:
        json_string = json.load(f)

    processed_samples = []

    samplenumber = 0
    for sample in json_string:
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
        processed_samples.append((dgl_graph, graph_to_token_alignments, token_to_char_alignments))

    return processed_samples


if __name__ == '__main__':
    f_name = 'amr_data_cola_train.json'
    graph_and_alignments = process(f_name)
    # print(graph_and_alignments)
    # Store data (serialize)
    # pickle code from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
    with open(f'{f_name}.pickle', 'wb') as handle:
        pickle.dump(graph_and_alignments, handle, protocol=pickle.HIGHEST_PROTOCOL)
