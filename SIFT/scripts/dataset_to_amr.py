import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from tqdm import tqdm
from data_readers import processors
from datasets import load_dataset
import amrlib


class SemanticEntailmentAMRDataset:
    def __init__(self, dataset_name, amr_parser_path='/home/david/tmp/model_parse_xfm_bart_base-v0_1_0'):

        self.amr_parser = amrlib.load_stog_model(amr_parser_path)
        self.dataset_name = dataset_name

        if dataset_name == 'hans':
            self.dataset = load_dataset('hans')
        elif dataset_name in ['mnli', 'wnli', 'qnli', 'mnli_mismatched', 'mnli_matched','cola']:
            self.dataset = load_dataset('glue', dataset_name)
        else:
            raise Exception(f'Dataset {dataset_name} does not exist.')

    def splits(self):
        return self.dataset.keys()

    def to_amr(self, split):
        if self.dataset_name != 'cola':
            raise Exception(f'Dataset {self.dataset_name} not supported.')
        sent_idx_list = [(entry['sentence'], entry['idx']) for entry in  self.dataset[split]]
        sent_list, idx_list = list(zip(*sent_idx_list))
        sent_list, idx_list = list(sent_list), list(idx_list)
        # print(sent_list, idx_list)
        parsed_sents = self.amr_parser.parse_sents(sent_list, add_metadata=True)
        return parsed_sents, idx_list


    def __str__(self):
        return str(self.dataset)






def convert(task, input_dir, output_dir):
    if any(os.path.isfile(os.path.join(output_dir, filename)) for filename in (TRAIN_FILE, DEV_FILE, TEST_FILE)):
        raise ValueError('Output file already exists.')

    processor = processors[task]()
    print(processor)

    # ugly code duplication
    if task == 'hans':
        train_examples = processor.get_train_examples(input_dir)
        dev_examples = processor.get_dev_examples(input_dir)

        for examples, filename in ((train_examples, TRAIN_FILE), (dev_examples, DEV_FILE)):
            with open(os.path.join(output_dir, filename), 'w') as f:
                for ex_index, example in enumerate(tqdm(examples)):
                    f.write(f'{ex_index * 2}\t{example.text_a}\n')
                    f.write(f'{ex_index * 2 + 1}\t{example.text_b}\n')
    else:
        train_examples = processor.get_train_examples(input_dir)
        dev_examples = processor.get_dev_examples(input_dir)
        test_examples = processor.get_test_examples(input_dir)

        for examples, filename in ((train_examples, TRAIN_FILE), (dev_examples, DEV_FILE), (test_examples, TEST_FILE)):
            with open(os.path.join(output_dir, filename), 'w') as f:
                for ex_index, example in enumerate(tqdm(examples)):
                    if example.text_b is None:
                        f.write(f'{ex_index}\t{example.text_a}\n')
                    else:
                        f.write(f'{ex_index * 2}\t{example.text_a}\n')
                        f.write(f'{ex_index * 2 + 1}\t{example.text_b}\n')


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
