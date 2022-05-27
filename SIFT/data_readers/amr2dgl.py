# some methods/classes adpated from https://github.com/dmlc/dgl/blob/ca48787ae49e9d73ebf17b1c882cb66fb66fb828/python/dgl/contrib/data/knowledge_graph.py

from collections import Counter
from dataclasses import dataclass
import pickle
import sys
from typing import Dict, Optional

from dgl import DGLGraph
import numpy as np
import torch
from tqdm import tqdm
import penman


# TODO: implement a small version of this to make stuff work
class AMRReader(object):
    __graph = None
    __freq = {}

    def __init__(self, graph):
        self.__graph = graph
        self.__freq = Counter(self.__graph.predicates())

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequency
        :return:
        """
        # TODO: change this to work for our case
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, rel):
        if rel not in self.__freq:
            return 0
        return self.__freq[rel]

# TODO: assume we already create the amr_graphs somewhere earlier
def amr2dgl(amr_graph, relation2id, bidirectional=True):

    p_graph = penman.decode(amr_graph)

    print(f"amr2dgl - penman metadata:   {p_graph.metadata}\n")
    print(f"amr2dgl - penman triples:   {p_graph.triples}\n")
    print(f"amr2dgl - penman edges:   {p_graph.edges}\n")
    print(f"amr2dgl - penman attributes:   {p_graph.attributes()}\n")
    # print(a.top, '\n\n')
    # print(a.epidata, '\n\n')

    nodes = set()
    edge_types = set()

    for edge in p_graph.edges():
        source, role, target = edge.source, edge.role, edge.target
        nodes.add(source)
        nodes.add(target)
        # v stands for vertex - not sure if we need multiple vertices in the future.
        edge_types.add(('v', role, 'v'))

    for attribute in a.attributes():
        source, role, target = attribute.source, attribute.role, attribute.target
        nodes.add(source)
        nodes.add(target)
        edge_types.add(('v', role, 'v'))

    print(nodes, '\n\n')
    print(edge_types, '\n\n')

    edge_collection = {}

    for edge_type in edge_types:
        # v1, t, v2  = edge_type
        edge_collection[edge_type] = []

    node_to_index = dict(list(map(lambda j: (j[1], j[0]), enumerate(list(nodes)))))

    print(node_to_index)

    for edge in a.edges():
        source, role, target = edge.source, edge.role, edge.target
        edge_collection[('v', role, 'v')].append((node_to_index[source], node_to_index[target]))

    for attribute in a.attributes():
        source, role, target = attribute.source, attribute.role, attribute.target
        edge_collection[('v', role, 'v')].append((node_to_index[source], node_to_index[target]))

    dgl_graph = {}

    for edge_type in edge_collection:
        u, v = list(zip(*edge_collection[edge_type]))
        u, v = list(u), list(v)
        dgl_graph[edge_type] = torch.tensor(u), torch.tensor(v)

    print('\n\n', dgl_graph, '\n\n')

    g = dgl.heterograph(dgl_graph)

    print(g)

    assert set(relation2id.values()) == set(range(len(relation2id)))

    with RDFReader(amr_graph) as reader:
        relations = reader.relationList()
        subjects = reader.subjectSet()
        objects = reader.objectSet()

        nodes = sorted(list(subjects.union(objects)))
        assert [int(node) for node in nodes] == list(range(len(nodes)))  # to make sure the metadata-node alignment is correct
        num_node = len(nodes)
        assert num_node == len(metadata)
        num_rel = len(relations)
        num_rel = 2 * num_rel # * 2 for bi-directionality

        if num_node == 0:
            g = DGLGraph()
            g.gdata = {'metadata': metadata}
            return g

        assert num_node < np.iinfo(np.int32).max

        edge_list = []

        for i, (s, p, o) in enumerate(reader.triples()):
            assert int(s) < num_node and int(o) < num_node
            rel = relation2id[p]
            edge_list.append((int(s), int(o), rel))
            if bidirectional:
                edge_list.append((int(o), int(s), rel + len(relation2id)))

        # sort indices by destination
        edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
        edge_list = np.array(edge_list, dtype=np.int)

    edge_src, edge_dst, edge_type = edge_list.transpose()

    # normalize by dst degree
    _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)

    node_ids = torch.arange(0, num_node, dtype=torch.long).view(-1, 1)
    edge_type = torch.from_numpy(edge_type)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)

    g = DGLGraph()
    g.add_nodes(num_node)
    g.add_edges(edge_src, edge_dst)
    g.ndata.update({'id': node_ids})
    g.edata.update({'type': edge_type, 'norm': edge_norm})

    g.gdata = {'metadata': metadata}  # we add this field in DGLGraph

    return g

# TODO: implement our version for the amr case
def amr_relations_in(rdf_graphs):

    raise NotImplementedError

    all_relations = set()
    for rdf_graph in tqdm(rdf_graphs):
        with AMRReader(rdf_graph) as reader:
            all_relations |= set(reader.relationList())
    return all_relations
