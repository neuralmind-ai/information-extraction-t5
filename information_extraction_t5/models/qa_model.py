"""Model definition based on Pytorh-Lightning."""
import os
import json
import configargparse
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from deepspeed.ops.adam import DeepSpeedCPUAdam

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    MT5ForConditionalGeneration,
    MT5Config
)

from information_extraction_t5.features.postprocess import (
    group_qas,
    get_highest_probability_window,
    split_compound_labels_and_predictions,
)
from information_extraction_t5.features.sentences import (
    get_clean_answer_from_subanswer
)
from information_extraction_t5.utils.metrics import (
    normalize_answer,
    t5_qa_evaluate,
    compute_exact,
    compute_f1
)
from information_extraction_t5.utils.freeze import freeze_embeds

class QAClassifier(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        if 'mt5' in self.hparams.config_name:
            config = MT5Config.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
            )
            self.model = MT5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=config,
                cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
            )
        else:
            config = T5Config.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=config,
                cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            do_lower_case=self.hparams.do_lower_case,
            use_fast=False,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )

        if 'byt5' in self.hparams.model_name_or_path.lower():
            self.input_max_length = self.hparams.max_size  # chars
        else:
            self.input_max_length = self.hparams.max_seq_length  # tokens

        # use for faster training/larger batch size
        freeze_embeds(self.model)

        # filename for cache predictions
        self.cache_fname = os.path.join(
            self.hparams.data_dir if self.hparams.data_dir else ".",
            "cached_predictions_{}.pkl".format(
                    list(filter(None, self.hparams.model_name_or_path.split("/"))).pop()
            )
        )

    def forward(self, x):
        return self.model(x)

class LitQA(QAClassifier, pl.LightningModule):
    
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer

    def training_step(self, batch, batch_idx):
        sentences, labels = batch

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )
        labels = self.tokenizer.batch_encode_plus(
            labels, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )

        inputs = {
            "input_ids": sentences_tokens['input_ids'].to(self.device),
            "labels": labels['input_ids'].to(self.device),
            "attention_mask": sentences_tokens['attention_mask'].to(self.device),
        }

        outputs = self.model(**inputs)

        self.log('train_loss', outputs[0], on_step=True, on_epoch=True,
            prog_bar=True, batch_size=len(sentences)
        )
        return {'loss': outputs[0]}

    def validation_step(self, batch, batch_idx):
        sentences, labels, _, _ = batch

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )

        inputs = {
            "input_ids": sentences_tokens['input_ids'].to(self.device),
            "attention_mask": sentences_tokens['attention_mask'].to(self.device),
            "max_length": self.hparams.max_length,
        }

        outputs = self.model.generate(**inputs)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'labels': labels, 'preds': predictions}

    def test_step(self, batch, batch_idx):
        sentences, labels, document_ids, typename_ids = batch

        # if we are using cached predictions, is not necessary to run steps again
        if self.hparams.use_cached_predictions and os.path.exists(self.cache_fname):
            return {'labels': [], 'preds': [], 'doc_ids': [], 'tn_ids': [], 'probs': []}

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )

        # This is handled differently then the others because of conflicts of
        # the previous approach with quantization.
        inputs = {
            "input_ids": sentences_tokens['input_ids'].to(self.device).long(),
            "attention_mask": sentences_tokens['attention_mask'].to(self.device).long(),
            "max_length": self.hparams.max_length,
            "num_beams": self.hparams.num_beams,
            "early_stopping": True,
        }

        outputs = self.model.generate(**inputs)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # compute probs
        probs = self._compute_probs(sentences, predictions)

        return {
            'labels': labels, 'preds': predictions, 'doc_ids': document_ids,
            'tn_ids': typename_ids, 'probs': probs
        }

    def validation_epoch_end(self, outputs):
        predictions, labels = [], []
        for output in outputs:
            for label, pred in zip(output['labels'], output['preds']):
                predictions.append(pred)
                labels.append(label)

        results = t5_qa_evaluate(labels, predictions)
        exact = torch.tensor(results['exact'])
        f1 = torch.tensor(results['f1'])

        log = {
            'val_exact': exact,       # for monitoring checkpoint callback
            'val_f1': f1,             # for monitoring checkpoint callback
        }
        self.log_dict(log, logger=True, prog_bar=True, on_epoch=True)

    def test_epoch_end(self, outputs):
        predictions, labels, document_ids, typename_ids, probs, window_ids = \
            [], [], [], [], [], []

        for output in outputs:
            for label, pred, doc_id, tn_id, prob in zip(
                output['labels'], output['preds'], 
                output['doc_ids'], output['tn_ids'], output['probs']):
                predictions.append(pred)
                labels.append(label)
                document_ids.append(doc_id)
                typename_ids.append(tn_id)
                probs.append(prob)

        # cache labels, predictions, document_ids, typename_ids and probs
        # so that we can post-process without running again the test_steps
        if self.hparams.use_cached_predictions and os.path.exists(self.cache_fname):
            print(f'Loading predictions from cached file {self.cache_fname}')
            labels, predictions, document_ids, typename_ids, probs = \
                pd.read_pickle(self.cache_fname).T.values.tolist()
        else:
            self._backup_outputs(labels, predictions, document_ids, typename_ids, probs)
                
        # pick up the highest-probability prediction for each pair document-typename
        if self.hparams.get_highestprob_answer:
            (
                labels,
                predictions,
                document_ids,
                typename_ids,
                probs,
                window_ids,
            ) = get_highest_probability_window(
                labels,
                predictions,
                document_ids,
                typename_ids,
                probs,
                use_fewer_NA=True,
            )

        # split compound answers to get metrics to visualize and compute metrics for each subsentence
        if self.hparams.split_compound_answers:
            (
                labels,
                predictions,
                document_ids,
                typename_ids,
                probs,
                window_ids,
                _, 
                _,
                original_idx,
                disjoint_answer_idx_by_doc_class,
            ) = split_compound_labels_and_predictions(
                labels,
                predictions,
                document_ids,
                typename_ids,
                probs,
                window_ids,
		    )
        else:
            print('WARNING: We strongly recommend to set --split_compound_answers=True, '
                  'even for datasets without compound qas. This is useful to get metrics '
                  'for clean outputs (without sentence-IDs and raw-text).')
            original_idx = list(range(len(labels)))
            disjoint_answer_idx_by_doc_class = {}

        # for each typename_id or document_id, extract its indexes to get specific metrics
        if self.hparams.group_qas:
            qid_dict_by_typenames = group_qas(typename_ids, group_by_typenames=True)
            qid_dict_by_documents = group_qas(document_ids, group_by_typenames=False)
            qid_dict_by_typenames['ORIG'] = original_idx
            qid_dict_by_documents['ORIG'] = original_idx
        else:
            qid_dict_by_typenames = {'ORIG': original_idx}
            qid_dict_by_documents = {'ORIG': original_idx}

        # save labels and predictions
        self._save_outputs(
            labels, predictions, document_ids,
            probs, window_ids, qid_dict_by_typenames, 
            outputs_fname='outputs_by_typenames.txt',
            document_classes=list(disjoint_answer_idx_by_doc_class.keys())
        )
        self._save_outputs(
            labels, predictions, typename_ids, 
            probs, window_ids, qid_dict_by_documents,
            outputs_fname='outputs_by_documents.txt', 
            document_classes=list(disjoint_answer_idx_by_doc_class.keys())
        )

        # For each document class, include the indexes of individual qas, and 
        # of subsentences of compound qas. This is useful for fair comparison 
        # of experiments with compound qas and individual qas.
        # Also save the disjoint samples in Excel sheets.
        all_idx = []
        writer = pd.ExcelWriter('outputs_sheet_client.xlsx')
        for document_class, indices in disjoint_answer_idx_by_doc_class.items():
            qid_dict_by_typenames['DISJOINT_' + document_class] = indices
            qid_dict_by_documents['DISJOINT_' + document_class] = indices
            all_idx += indices
            self._save_sheets(
                labels, predictions, document_ids, 
                typename_ids, probs, document_class, indices, writer
            )
        writer.close()
        qid_dict_by_typenames['DISJOINT_ALL'] = all_idx
        qid_dict_by_documents['DISJOINT_ALL'] = all_idx
        self._save_sheets(
            labels, predictions, document_ids,
            typename_ids, probs, 'all', all_idx
        )

        # compute metrics
        results_by_typenames = t5_qa_evaluate(
            labels, predictions, qid_dict=qid_dict_by_typenames
        )
        results_by_documents = t5_qa_evaluate(
            labels, predictions, qid_dict=qid_dict_by_documents
        )
        exact = torch.tensor(results_by_typenames['exact'])
        f1 = torch.tensor(results_by_typenames['f1'])

        # write metric files
        with open('metrics_by_typenames.json', 'w') as f:
            json.dump(results_by_typenames, f, indent=4)
        with open('metrics_by_documents.json', 'w') as f:
            json.dump(results_by_documents, f, indent=4)

        log = {
            'exact': exact,
            'f1': f1
        }
        self.log_dict(log, logger=True, on_epoch=True)

    @torch.no_grad()
    def _compute_probs(self, sentences, predictions):
        probs = []
        for sentence, prediction in zip(sentences, predictions):
            input_ids = self.tokenizer.encode(sentence, truncation=True, 
                max_length=self.input_max_length, return_tensors="pt").to(self.device).long()
            output_ids = self.tokenizer.encode(prediction, truncation=True, 
                max_length=self.input_max_length, return_tensors="pt").to(self.device).long()

            outputs = self.model(input_ids=input_ids, labels=output_ids)

            loss = outputs[0]
            prob = (loss * -1) / output_ids.shape[1]
            prob = np.exp(prob.cpu().numpy())
            probs.append(prob)
        return probs

    def _backup_outputs(self, labels, predictions, document_ids, typename_ids, probs):
            arr = np.vstack([np.array(labels, dtype="O"), np.array(predictions, dtype="O"),
                            np.array(document_ids, dtype="O"), np.array(typename_ids, dtype="O"),
                            np.array(probs, dtype="O")]).transpose()
            df = pd.DataFrame(arr, columns=['labels', 'predictions', 'document_ids', 'typename_ids', 'probs'])
            df.to_pickle(self.cache_fname)

    def _save_outputs(
        self, labels, predictions, doc_or_tn_ids, probs, window_ids,
        qid_dict=None, outputs_fname='outputs.txt', document_classes=["form"]
        ):
        if qid_dict is None:
            qid_dict = {}

        f = open(outputs_fname, 'w')
        f.write('{0:<50} | {1:50} | {2:30} | {3} | {4}\n'.format(
            'label', 'prediction', 'uuid', 'prob', 'window'))
        if qid_dict == {}:
            for label, prediction, doc_or_tn_id, prob, w_id in zip(
                labels, predictions, doc_or_tn_ids, probs, window_ids):
                lab, pred = label, prediction
                if self.hparams.normalize_outputs:
                    lab, pred = normalize_answer(label), normalize_answer(prediction)
                if lab != pred or lab == pred and not self.hparams.only_misprediction_outputs:
                    f.write('{0:<50} | {1:50} | {2:30} | {3} | {4}\n'.format(
                        label, prediction, doc_or_tn_id, prob, w_id))
        else:
            for (kword, list_indices) in qid_dict.items():
                # do not print for ORIG, DISJOINT* and all samples for a specific project/document class
                # those groups are important for metrics, not for outputs visualization
                if kword == 'ORIG' or kword.startswith('DISJOINT') or kword in document_classes: 
                    continue
                f.write(f'===============\n{kword}\n===============\n')
                for idx in list_indices:
                    label, prediction, doc_or_tn_id, prob, w_id = \
                        labels[idx], predictions[idx], doc_or_tn_ids[idx], probs[idx], window_ids[idx]
                    lab, pred = label, prediction
                    if self.hparams.normalize_outputs:
                        lab, pred = normalize_answer(label), normalize_answer(prediction)
                    if lab != pred or lab == pred and not self.hparams.only_misprediction_outputs:
                        f.write('{0:<50} | {1:50} | {2:30} | {3} | {4}\n'.format(
                            label, prediction, doc_or_tn_id, prob, w_id))
        f.close()

    def _save_sheets(self, labels, predictions, document_ids, typename_ids, probs, document_class, indices, writer=None):
        # Saving disjoint predictions (splitted and clean) in a dataframe
        arr = np.vstack([np.array(document_ids, dtype="O")[indices],
                        np.array(typename_ids, dtype="O")[indices],
                        np.array(labels, dtype="O")[indices],
                        np.array(predictions, dtype="O")[indices],
                        np.array(probs, dtype="O")[indices]]).transpose()
        df = pd.DataFrame(arr, 
            columns=['document_ids', 'typename_ids', 'labels', 'predictions', 'probs']
            ).reset_index(drop=True)

        if document_class == 'all':
            df = df.sort_values(['document_ids', 'typename_ids'])  # hack to keep listing outputs together for each document-class
            df_all_group_doc = df.set_index('document_ids', append=True).swaplevel(0,1)
            df_all_group_doc.to_excel('outputs_sheet.xlsx')
        else:
            # compute metrics for each pair document_id-typename-id
            df['exact'] = df.apply(lambda x: compute_exact(x['labels'], x['predictions']), axis=1)
            df['f1'] = df.apply(lambda x: compute_f1(x['labels'], x['predictions']), axis=1)

            # remove clue/prefix into brackets
            df['labels'] = df.apply(
                lambda x: ', '.join(get_clean_answer_from_subanswer(x['labels'])),
                axis=1
            )
            df['predictions'] = df.apply(
                lambda x: ', '.join(get_clean_answer_from_subanswer(x['predictions'])),
                axis=1
            )

            # use pivot to get a quadruple of columns (labels, predictions, equal, prob) for each typename
            pivoted = df.pivot(
                index=['document_ids'],
                columns=['typename_ids'],
                values=['labels', 'predictions', 'exact', 'f1', 'probs']
            )
            pivoted = pivoted.swaplevel(0, 1, axis=1).sort_index(axis=1)  # put column (typename_ids) above the values

            # extract typename_ids in the original order (instead of alphanumeric order)
            # get the columns from the document-ids that have more samples
            cols = df[df['document_ids']==df.document_ids.mode()[0]].typename_ids.tolist()
            if len(cols) == len(pivoted.columns) // 5:
                pivoted = pivoted[cols]
            else:
                print('Keeping typenames in alphanumeric order since none of the documents '
                    f'have all the possible qa_ids ({len(cols)} != {len(pivoted.columns) // 5})')

            # save sheet
            pivoted.to_excel(writer, sheet_name=document_class)

    def get_optimizer(self,) -> torch.optim.Optimizer:
        """Define the optimizer"""
        optimizer_name = self.hparams.optimizer
        lr = self.hparams.lr
        weight_decay=self.hparams.weight_decay
        optimizer = getattr(torch.optim, optimizer_name)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if self.hparams.deepspeed:
            # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters, lr=lr, 
                weight_decay=weight_decay, eps=1e-4, adamw_mode=True
            )
        else:
            optimizer = optimizer(
                optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay
            )

        print(f'=> Using {optimizer_name} optimizer')

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = configargparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--model_name_or_path",
            default='t5-small',
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
        )
        parser.add_argument(
            "--max_seq_length",
            default=384,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        parser.add_argument(
            "--max_size",
            default=1024,
            type=int,
            help="The maximum input length after char-based tokenization. And also the maximum context "
            "size for char-based contexts."
        )
        parser.add_argument(
            "--max_length",
            default=120,
            type=int,
            help="The maximum total output sequence length generated by the model."
        )
        parser.add_argument(
            "--num_beams",
            default=1,
            type=int,
            help="Number of beams for beam search. 1 means no beam search."
        )
        parser.add_argument(
            "--get_highestprob_answer", 
            action="store_true",
            help="If true, get the answer from the sliding-window that gives highest probability."
        )
        parser.add_argument(
            "--split_compound_answers",
            action="store_true",
            help="If true, split the T5 outputs into individual answers.",
        )
        parser.add_argument(
            "--group_qas",
            action="store_true",
            help="If true, use group qas to get individual metrics ans structured output file for each type-name.",
        )
        parser.add_argument(
            "--only_misprediction_outputs",
            action="store_true",
            help="If true, return only mispredictions in the output file.",
        )
        parser.add_argument(
            "--normalize_outputs",
            action="store_true",
            help="If true, normalize label and prediction to include in the output file. " 
                 "The normalization is the same applied before computing metrics.",
        )

        return parser
