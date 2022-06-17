combos = {
        'hans': ['train', 'validation'],
        'mnli': ['test_matched', 'test_mismatched', 'train', 'validation_mismatched', 'validation_matched'],
        'cola': ['test', 'train', 'validation'],
        'mnli_matched': ['test', 'train', 'validation'],
        'mnli_mismatched': ['test', 'train', 'validation'],
        'mrpc': ['test', 'train', 'validation'],
        'qnli': ['test', 'train', 'validation'],
        'sst2':  ['test', 'train', 'validation'],
        'qqp': ['test', 'train', 'validation'],
        'stsb': ['test', 'train', 'validation'],
        'wnli': ['test', 'train', 'validation'],
        'ax': ['test'],
        'rte': ['test', 'train', 'validation'],
    }

def copy_files(tasks):
    for dataset in tasks.keys():
        ds = SemanticEntailmentAMRDatahow set(dataset)
        for task in tasks[dataset]:
            res = ds.to_amr(task)
            f_name = f'amr_data_{dataset}_{task}.json'
            with open(f_name, 'w') as f:
                json.dump(res,f)
            print(f'Just finished {f_name}')




python amr_to_RDF.py amr_data_cola_train.json train.amr.rdf train.amr.metadata
python amr_to_RDF.py amr_data_cola_test.json test.amr.rdf test.amr.metadata
python amr_to_RDF.py amr_data_cola_validation.json dev.amr.rdf dev.amr.metadata
