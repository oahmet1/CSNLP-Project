import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from tqdm import tqdm

from data_readers import processors
from datasets import load_dataset

extension = '.lines'



def convert(task, output_dir):

    if task == 'hans':
        dataset = load_dataset('hans')
    elif task in ['mnli', 'wnli', 'qnli', 'mnli_mismatched', 'mnli_matched', 'cola', 'rte','mrpc', 'sst2', 'ax', 'qqp' ,'stsb']:
        dataset = load_dataset('glue', task)

    keys = dataset.keys()

    for key in keys:
        fname = key + extension
        gname = key + '.lnn'
        Path(os.path.join(output_dir, fname)).parent.mkdir(exist_ok=True, parents=True)
        with open(os.path.join(output_dir, fname), 'w') as f:
            with open(os.path.join(output_dir, gname), 'w') as g:
                for ex_index, example in enumerate(tqdm(dataset[key])):
                    if task  in ['sst2', 'cola']:
                        f.write(f'{ex_index}\t{example["sentence"]}\n')
                        g.write(f'{example["sentence"]}\n')

                    elif task in ['mnli', 'mnli_mismatched', 'mnli_matched', 'ax', 'hans']:
                        f.write(f'{ex_index * 2}\t{example["premise"]}\n')
                        f.write(f'{ex_index * 2 + 1}\t{example["hypothesis"]}\n')

                        g.write(f'{example["premise"]}\n')
                        g.write(f'{example["hypothesis"]}\n')

                    elif task in ['qnli']:
                        f.write(f'{ex_index * 2}\t{example["question"]}\n')
                        f.write(f'{ex_index * 2 + 1}\t{example["sentence"]}\n')

                        g.write(f'{example["question"]}\n')
                        g.write(f'{example["sentence"]}\n')

                    elif task in ['wnli', 'stsb', 'rte', 'mrpc']:
                        f.write(f'{ex_index * 2}\t{example["sentence1"]}\n')
                        f.write(f'{ex_index * 2 + 1}\t{example["sentence2"]}\n')

                        g.write(f'{example["sentence1"]}\n')
                        g.write(f'{example["sentence2"]}\n')

                    elif task in ['qqp']:
                        f.write(f'{ex_index * 2}\t{example["question1"]}\n')
                        f.write(f'{ex_index * 2 + 1}\t{example["question2"]}\n')

                        g.write(f'{example["question1"]}\n')
                        g.write(f'{example["question2"]}\n')



if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
