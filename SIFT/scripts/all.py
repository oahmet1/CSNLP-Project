import os
import glob
'''
os.system(f'rm -rf data/')
'''
task_dataset_combo = [('cola', 'data/glue_data/CoLA/', ['test', 'train', 'validation']),
                      ('mnli', 'data/glue_data/MNLI/', ['test_mismatched','test_matched', 'train', 'validation_matched', 'validation_mismatched']),
                      ('mrpc', 'data/glue_data/MRPC/', ['test', 'train', 'validation']),
                      ('qnli', 'data/glue_data/QNLI/', ['test', 'train', 'validation']),
                      ('qqp', 'data/glue_data/QQP/',  ['test', 'train', 'validation']),
                      ('rte', 'data/glue_data/RTE/',  ['test', 'train', 'validation']),
                      ('sst2', 'data/glue_data/SST-2/',  ['test', 'train', 'validation']),
                      ('stsb', 'data/glue_data/STS-B/',  ['test', 'train', 'validation']),
                      ('wnli', 'data/glue_data/WNLI/',  ['test', 'train', 'validation']),
                      ('hans', 'data/HANS/',  ['train', 'validation'])]

'''
for (TASK_NAME,DATASET_DIR, _) in task_dataset_combo:
    os.system(f'python scripts/dataset_to_CoNNL_U.py {TASK_NAME} {DATASET_DIR}')


all_lines_files = glob.glob('data/**/*.lnn', recursive=True)

for line_file in list(all_lines_files):
    print(f'Processing {line_file}')
    os.system(f'scripts/udpipe --tokenizer ranges --tag --parse  scripts/english-partut-ud-2.5-191206.udpipe {line_file} --outfile={os.path.splitext(line_file)[0]}.conllu')
'''
for (TASK_NAME,DATASET_DIR, SPLITS) in task_dataset_combo:
    for split in SPLITS:
        os.system(f'bash scripts/decode_che_et_al.sh {DATASET_DIR} {split}')



'''

2.
`${SPLIT}.lines` into `${SPLIT}.conllu`.

3.     bash scripts/decode_che_et_al.sh ${DATASET_DIR}

'''