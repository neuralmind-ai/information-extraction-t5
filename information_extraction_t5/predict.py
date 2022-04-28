# coding=utf-8
""" Predicting the T5 model finetuned for question-answering on SQuAD."""

import configargparse
import glob

import torch
from pytorch_lightning import Trainer

from information_extraction_t5.models.qa_model import LitQA
from information_extraction_t5.data.qa_data import QADataModule


def main():
    """Predict."""

    parser = configargparse.ArgParser(
        'Training and evaluation script for training T5 model for QA', 
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True,
        help='config file path')

    parser.add_argument("--seed", type=int, default=42,
        help="random seed for initialization")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--use_cached_predictions", action="store_true",
        help="If true, reload the cache to post-process the senteces and compute metrics")

    parser = LitQA.add_model_specific_args(parser)
    parser = QADataModule.add_model_specific_args(parser)
    args, _ = parser.parse_known_args()

    # Load best checkpoint of the current experiment
    ckpt_path = glob.glob('lightning_logs/*ckpt')[0]
    print(f'Loading weights from {ckpt_path}')

    model = LitQA.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file='lightning_logs/version_0/hparams.yaml',
        map_location=None,
        hparams=args,
    )
    gpus = 1 if torch.cuda.is_available() else 0
    if gpus == 0:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    dm = QADataModule(args)
    dm.setup('test')

    torch.set_num_threads(1)
    trainer = Trainer(gpus=gpus)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
