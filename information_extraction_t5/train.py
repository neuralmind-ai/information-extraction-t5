# coding=utf-8
""" Finetuning the T5 model for question-answering on SQuAD."""

import math
import os
import configargparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.plugins import DeepSpeedPlugin

from information_extraction_t5.models.qa_model import LitQA
from information_extraction_t5.data.qa_data import QADataModule

MODEL_DIR = 'lightning_logs'

def main():
    """Train."""

    parser = configargparse.ArgParser(
        'Training and evaluation script for training T5 model for QA',
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True,
        help='config file path')

    # optimizer parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument("--lr", default=5e-5, type=float,
        help="The initial learning rate.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
        help="Weight decay if we apply some.")

    # neptune
    parser.add_argument("--neptune", action="store_true", help="If true, use neptune logger.")
    parser.add_argument('--neptune_project', type=str, default='ramon.pires/bracis-2021')
    parser.add_argument('--experiment_name', type=str, default='experiment01')
    parser.add_argument('--tags', action='append')

    parser.add_argument("--deepspeed", action="store_true", help="If true, use deepspeed plugin.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--num_workers', default=8, type=int)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    # add model specific args
    parser = LitQA.add_model_specific_args(parser)
    # add datamodule specific args
    parser = QADataModule.add_model_specific_args(parser) 
    args, _ = parser.parse_known_args()

    # cached predictions must be used only for predict.py
    args.use_cached_predictions = False

    # setting the seed for reproducibility
    if args.deterministic:
        seed_everything(args.seed)

    # data module
    dm = QADataModule(args)
    dm.setup('fit')

    # Defining the model
    model = LitQA(args)

    # For training larger models, we are not running validation but saving
    # checkpoint by train steps.
    # To doing this, we set check_val_every_n_epoch > max_epochs
    if args.check_val_every_n_epoch > args.max_epochs:
        # dataset_size / (batch_size * accum_batches)
        every_n_train_steps = math.ceil(
            len(dm.train_dataset) / (args.train_batch_size * args.accumulate_grad_batches))
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODEL_DIR, filename='{epoch}-{train_loss:.4f}',
            monitor='train_loss_step', verbose=False, save_last=False, save_top_k=args.max_epochs,
            save_weights_only=True, mode='min', every_n_train_steps=every_n_train_steps
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODEL_DIR, filename='{epoch}-{val_exact:.2f}-{val_f1:.2f}',
            monitor='val_exact', verbose=False, save_last=False, save_top_k=5,
            save_weights_only=False, mode='max', every_n_epochs=1
        )

    # Instantiate LearningRateMonitor Callback
    lr_logger = LearningRateMonitor(logging_interval='epoch')

    # Set neptune logger
    if args.neptune:
        neptune_logger = NeptuneLogger(
            api_key=os.environ.get('NEPTUNE_API_TOKEN'),
            project=args.neptune_project,
            name=args.experiment_name,
            mode='async',  # Possible values "async", "sync", "offline", and "debug", "read-only"
            run=None,  # Set run's identifier like 'SAN-1' in case of resuming a tracked run
            tags=args.tags,
            log_model_checkpoints=False,
            source_files=["**/*.py", "*.yaml"],
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
        )
    else:
        neptune_logger = None

    if args.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            stage=2,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=2e8,
            reduce_bucket_size=2e8,
            allgather_partitions=True,
            reduce_scatter=True,
            overlap_comm=True,
            contiguous_gradients=True,
            ## Activation Checkpointing
            partition_activations=True,
            cpu_checkpointing=True,
            contiguous_memory_optimization=True,
        )
    else:
        deepspeed_plugin = None

    # Defining the Trainer, training... and finally testing
    trainer = Trainer.from_argparse_args(
        args,
        logger=neptune_logger,
        plugins=deepspeed_plugin,
        callbacks=[
            lr_logger,
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ]
    )
    trainer.fit(model, datamodule=dm)

    dm.setup('test')
    trainer.test(datamodule=dm)

    # Save checkpoints folder
    if args.neptune:
        # neptune_logger.experiment.log_artifact(MODEL_DIR)
        neptune_logger.log_hyperparams(vars(args))
        neptune_logger.run['training/artifacts/metrics_by_typenames.json'].log('metrics_by_typenames.json')
        neptune_logger.run['training/artifacts/metrics_by_documents.json'].log('metrics_by_documents.json')
        neptune_logger.run['training/artifacts/outputs_sheet_client.xlsx'].log('outputs_sheet_client.xlsx')


if __name__ == "__main__":
    main()
