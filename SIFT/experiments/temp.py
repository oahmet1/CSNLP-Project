from datasets import load_dataset
import amrlib
from amrlib.alignments.rbw_aligner import RBWAligner
from amrlib.graph_processing.annotator import add_lemmas
import spacy



class SemanticEntailmentAMRDataset:
    def __init__(self, dataset_name, amr_parser_path='/home/david/tmp/model_parse_xfm_bart_base-v0_1_0'):

        self.amr_parser = amrlib.load_stog_model(amr_parser_path)
        self.dataset_name = dataset_name
        self.nlp_tokenizer = spacy.load("en_core_web_sm")

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
        # TODO does parse_sents reverse order again ???
        return parsed_sents, idx_list

    def aligned_AMR(self, arm_graph_string):
        pg = add_lemmas(arm_graph_string, snt_key='snt')
        aligner = RBWAligner.from_penman_w_json(pg)
        penman_graph = aligner.get_penman_graph()
        return penman_graph


    def get_alignments(self, text):
        # returns an array where the i-th entry is a 2-tuple of the start and end characters index of the word (the last exclusive!)
        # TODO inclusive or exclusive?
        tokenized = self.nlp_tokenizer(text)
        alignments = []
        for i, token in enumerate(tokenized):
            alignments.append((token.idx, token.idx + len(token) - 1))
        return alignments

    def __str__(self):
        return str(self.dataset)



[(penman_dings, alginment_list),(penman_dings, alginment_list),(penman_dings, alginment_list),(penman_dings, alginment_list)]



a = SemanticEntailmentAMRDataset('cola')

print(a)
