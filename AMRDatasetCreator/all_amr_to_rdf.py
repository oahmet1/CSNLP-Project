import glob
from pathlib import Path
import os

dataset_lookup_table = {
    'cola': 'CoLA',
    'sst2': 'SST-2',
    'sstb': 'SST-B'
}

all_json_files = list(glob.glob('*.pkl'))

for json_file in all_json_files:
    json_path = Path(json_file)
    splitted_stem = json_path.stem.split('_')
    dataset, split = splitted_stem[2], splitted_stem[3]

    if dataset == 'mnli':
        split = splitted_stem[4]
        if splitted_stem[3] in ['mismatched', 'matched']:
            split = split + '2'

    if split == 'validation':
        split = 'dev'

    print(dataset, split)

    dataset = dataset_lookup_table[dataset] if dataset in dataset_lookup_table.keys() else dataset.upper()
    os.makedirs(f'../amr_rdf_graphs/{dataset}/', exist_ok=True)

    os.system(f'python amr_to_RDF.py {json_path.name} ../amr_rdf_graphs/{dataset}/{split}.amr.rdf ../amr_rdf_graphs/{dataset}/{split}.amr.metadata')
    # os.system(f'rsync -av --progress ../amr_rdf_graphs/ ../SIFT/data/glue_data/ --dry-run')


print('finished')
