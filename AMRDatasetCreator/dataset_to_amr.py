from datasets import load_dataset
import amrlib
from amrlib.alignments.rbw_aligner import RBWAligner
from amrlib.alignments.faa_aligner import FAA_Aligner
from amrlib.graph_processing.annotator import add_lemmas
import spacy
import penman
import json
import os
import sys


class SemanticEntailmentAMRDataset:
    def __init__(self, dataset_name, amr_parser_path='/cluster/scratch/zdavid/models/model_parse_xfm_bart_base-v0_1_0'):
        self.amr_parser = amrlib.load_stog_model(amr_parser_path)
        self.dataset_name = dataset_name
        self.nlp_tokenizer = spacy.load('en_core_web_sm')

        if dataset_name == 'hans':
            self.dataset = load_dataset('hans')
        elif dataset_name in ['mnli', 'wnli', 'qnli', 'mnli_mismatched', 'mnli_matched', 'cola', 'rte','mrpc', 'sst2', 'ax', 'qqp' ,'stsb']:
            self.dataset = load_dataset('glue', dataset_name)
        else:
            raise Exception(f'Dataset {dataset_name} does not exist.')

    def splits(self):
        return self.dataset.keys()

    def to_amr(self, split):
        if self.dataset_name  in ['sst2', 'cola']:
            sent_idx_list = [(entry['sentence'], entry['idx']) for entry in  self.dataset[split]]
            sent_list, idx_list = list(zip(*sent_idx_list))
            sent_list, idx_list = list(sent_list), list(idx_list)
            # print(sent_list, idx_list)

        elif self.dataset_name in ['mnli', 'mnli_mismatched', 'mnli_matched', 'ax', 'hans']:
            sent_list = []
            for entry in self.dataset[split]:
                sent_list.extend([entry['premise'], entry['hypothesis']])

        elif self.dataset_name in ['qnli']:
            sent_list = []
            for entry in self.dataset[split]:
                sent_list.extend([entry['question'], entry['sentence']])

        elif self.dataset_name in ['wnli', 'stsb', 'rte', 'mrpc']:
            sent_list = []
            for entry in self.dataset[split]:
                sent_list.extend([entry['sentence1'], entry['sentence2']])

        elif self.dataset_name in ['qqp']:
            sent_list = []
            for entry in self.dataset[split]:
                sent_list.extend([entry['question1'], entry['question2']])
        else:
            raise Exception(f'Dataset {self.dataset_name} not supported.')

        print('Dataset loaded, starting AMR parsing.')
        parsed_sents = self.amr_parser.parse_sents(sent_list, add_metadata=True)
        print('All parsed to AMR')

        aligned_amr_sentences = []
        for idx, amr_sentence in enumerate(parsed_sents):
            # print(amr_sentence, sent_list[idx])
            aligned_amr_sentences.append(self.__aligned_AMR(amr_sentence, sent_list[idx]))

        print('All AMR aligned')

        processed_amr_sentence_alignment_pairs = []
        for aligned_amr_sentence in aligned_amr_sentences:
            processed_amr_sentence_alignment_pairs.append((penman.encode(aligned_amr_sentence),self.__get_alignments(aligned_amr_sentence)))

        print('All Tokens aligned')

        return processed_amr_sentence_alignment_pairs

    def __aligned_AMR(self, amr_graph_string, sentence):
        pg = add_lemmas(amr_graph_string, snt_key='snt')
        aligner = RBWAligner.from_penman_w_json(pg)
        penman_graph = aligner.get_penman_graph()
        return penman_graph


    def __get_alignments(self, penman_graph):
        # returns an array where the i-th entry is a 2-tuple of the start and end characters index of the word (the last exclusive!)
        alignments = []
        ref_token = self.nlp_tokenizer(penman_graph.metadata['snt'])
        for i, token in enumerate(json.loads(penman_graph.metadata['tokens'])):
            assert token == ref_token[i].text, f'token are unexpectedly different {token} - {ref_token[i].text}'
            alignments.append((ref_token[i].idx, ref_token[i].idx + len(ref_token[i])))
        return alignments

    def __str__(self):
        return str(self.dataset)





def preprocess_all_data(tasks):
    for dataset in tasks.keys():
        ds = SemanticEntailmentAMRDataset(dataset)
        for task in tasks[dataset]:
            res = ds.to_amr(task)
            f_name = f'amr_data_{dataset}_{task}.json'
            with open(f_name, 'w') as f:
                json.dump(res,f)
            print(f'Just finished {f_name}')



if __name__ == '__main__':

    # enter all tasks that should be prepared
    combos = {
        'hans': ['train', 'validation'],
        'mnli': ['test_matched', 'test_mismatched', 'train', 'validation_mismatched', 'validation_matched'],
        'cola': ['test', 'train', 'validation'],
        'mnli_matched': ['test', 'validation'],
        'mnli_mismatched': ['test', 'validation'],
        'mrpc': ['test', 'train', 'validation'],
        'qnli': ['test', 'train', 'validation'],
        'sst2':  ['test', 'train', 'validation'],
        'qqp': ['test', 'train', 'validation'],
        'stsb': ['test', 'train', 'validation'],
        'wnli': ['test', 'train', 'validation'],
        'ax': ['test'],
        'rte': ['test', 'train', 'validation'],
    }

    task_dict = {sys.argv[1]: combos[sys.argv[1]]}
    preprocess_all_data(task_dict)




