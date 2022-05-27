
#### This file is just to ensure everyone is on the same page and to let each other know about assumptions/plans


- we use the formalism name "amr"
- 
- 
- store the amr graphs in pickle files of the form: ??  pickle.load(open(os.path.join(data_dir, f'train.{formalism}.rdf'), 'rb')) ??
- then we read them in the base_reader, in the get_graphs function
- the get_graphs function is called in the semantic_encoder in prepare_data()

-  need to figure out where/how exactly we get the correct indices to initialize the graph nodes with embeddings!
    - see pool_node_embeddings in semantic_encoder, but we need to adapt this!
