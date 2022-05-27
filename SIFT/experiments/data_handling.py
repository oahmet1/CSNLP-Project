from datasets import load_dataset
import amrlib

class SemanticEntailmentAMRDataset:
    def __init__(self, dataset, amr_parser_path='/home/david/tmp/model_parse_xfm_bart_base-v0_1_0'):

        self.amr_parser = amrlib.load_stog_model(amr_parser_path)

        if dataset == 'hans':
            self.dataset = load_dataset('hans')
        elif dataset in ['mnli', 'wnli', 'qnli', 'mnli_mismatched', 'mnli_matched']:
            self.dataset = load_dataset('glue', dataset)
        else:
            raise Exception(f'Dataset {dataset} does not exist.')

    def __str__(self):
        return str(self.dataset)


a = SemanticEntailmentAMRDataset('qnli')

print(a)
