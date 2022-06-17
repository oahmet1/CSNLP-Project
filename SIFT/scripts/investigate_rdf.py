import pickle
from rdflib import Graph



l1 = pickle.load(open('../data/glue_data/CoLA/dev.dm.rdf','rb'))

l2 = pickle.load(open('../data/glue_data/CoLA/dev.dm.metadata','rb'))

print(l1[0])
print(l2[0])

s = l1[0].serialize(format="turtle").decode('ascii')

print(str(s))

# g = Graph()
# g.parse(l1)
