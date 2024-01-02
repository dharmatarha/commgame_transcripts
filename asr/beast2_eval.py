"""
Slightly edited version of the original "eval_asr_lm.py" script from the BEAST2 repository.
BEAST2 available from: https://phon.nytud.hu/bea/bea-base.html?lang=hu

Runs ASR on preprocessed / prepared audio.

USAGE:
python3 beast2_eval.py PAIR_NUMBERS --data_dir DATA_DIR_PATH

"""


import logging
import random
import sys
import torch
import time
import torchaudio

from speechbrain.lobes.models.transformer import TransformerLM
from speechbrain.tokenizers.SentencePiece import SentencePiece
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import undo_padding

import os
import argparse
import speechbrain as sb
import sentencepiece as spm
import numpy as np
from glob import glob


class ASR(sb.core.Brain):

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.modules.wav2vec2(wavs, wav_lens)
        x = self.modules.enc(feats)

        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
        return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        loss = loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # Here we manage mixed precision
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                predictions = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        else:
            predictions = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # loss.backward()
        (loss / self.hparams.accu_steps).backward()

        if self.step % self.hparams.accu_steps == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.wav2vec_optimizer.step()
            self.model_optimizer.step()
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


def dataio_prepare(hparams):
    data_folder = hparams["data_folder"]
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"], replacements={"data_root": data_folder},
    )

    test_data = test_data.filtered_sorted(sort_key="length")

    datasets = [test_data]

    tokenizer = SentencePiece(model_dir=hparams["tokenizer_path"],
                              vocab_size=hparams["output_neurons"])

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("words")  # we have our own data format, not "wrd"
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return test_data, tokenizer


def main():
    # Input argument handling
    # One mandatory and one optional argument, defining pair number(s) and the data folder which contains
    # beast2 input yaml files.
    # Arg args.pair_numbers is a list with one or more elements while args.data_dir is string (path to dir).
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair numbers, determines which audio (wav) files are selected for analysis.'
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    parser.add_argument('--data_dir', type=str, default=os.getcwd(), nargs='?',
                        help='Path to directory holding the beast2 param (yaml) files.')
    parser.add_argument('--use_filtered', action='store_true',
                        help='Flag for using rms-filtered audio as input, that is, '
                             'preprocessed files with "_filtered" in their names.')
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.data_dir):
        raise ValueError('Input arg --data_dir is not a valid path!')

    # If there are two pair numbers, expand it into a range
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers

    print('\n\nCalled beast2_eval with args:')
    print('Pair numbers: ' + str(pair_numbers))
    print('Data folder: ' + args.data_dir)
    print('Filtered data flag: ' + str(args.use_filtered))

    print('\n\nLooping through pairs...')

    # Loop through pairs
    for pair in pair_numbers:

        print('\n\n\nPAIR', pair)

        # Find all yaml files for pair
        if args.use_filtered:
            yaml_pattern = os.path.join(args.data_dir, '**', 'pair' + str(pair)
                                        + '_*dor_*_repaired_mono_noisered_filtered_hparams.yaml')
        else:
            yaml_pattern = os.path.join(args.data_dir, '**', 'pair' + str(pair)
                                        + '_*dor_*_repaired_mono_noisered_hparams.yaml')
        pair_yaml_files = glob(yaml_pattern, recursive=True)
        if len(pair_yaml_files) == 0:
            raise FileNotFoundError('Could not find any beast2 param file for pair', pair, '!!!')
        else:
            print('\nFound', len(pair_yaml_files), 'files for pair', pair)
            for file_path in pair_yaml_files:
                print(file_path)
            print('Looping through yaml files...')

        # Loop through beast2 param (yaml) files, init ASR for each
        for yaml_file in pair_yaml_files:

            print('\nWorking on', yaml_file)

            hparams_file, run_opts, overrides = sb.parse_arguments([yaml_file, '', ''])
            with open(hparams_file) as fin:
                hparams = load_hyperpyyaml(fin, overrides)

            run_on_main(hparams["pretrainer"].collect_files)
            hparams["pretrainer"].load_collected(device=run_opts["device"])

            test_data, tokenizer = dataio_prepare(hparams)

            asr_brain = ASR(
                modules=hparams["asr_modules"],
                run_opts=run_opts,
                hparams=hparams,
            )

            asr_brain.tokenizer = tokenizer

            asr_brain.evaluate(
                test_data,
                min_key="WER",
                test_loader_kwargs=hparams["test_dataloader_opts"]
            )


if __name__ == "__main__":
    main()
