import os 



os.system('pip install spacy penman word2number unidecode transformers datasets')
os.system('python -m spacy download en_core_web_sm')

from datasets import load_dataset


glue = {
    'mnli',
    'cola',
    'mnli_matched',
    'mnli_mismatched',
    'mrpc',
    'qnli',
    'sst2',
    'qqp',
    'stsb',
    'wnli',
    'ax',
    'rte'
}

_ = load_dataset('hans')
for glue_task in glue:
    _ = load_dataset('glue', glue_task)


