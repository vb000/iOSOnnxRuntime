import os
import time
import sys
import importlib
import logging
import random
import argparse
import json

import numpy as np
import onnx
import torch
import onnxruntime as ort
import torch.nn as nn

from src.helpers import utils

OPSET=15

def log_allclose(gt_output, output, rtol=1e-05, atol=1e-05):
    if torch.allclose(gt_output, output, rtol=rtol, atol=atol):
        logging.info("Test ONNX success.")
    else:
        logging.info("Test ONNX error.")

def save_onnx_parallel(out_dir, test=False):
    """ Converts a model to onnx file """
    # Load model and training params
    params = utils.Params('experiments/parallel_waveformer_128_snr/config.json')
    onnx_file = os.path.join(out_dir, 'model.onnx')
    log_file = os.path.join(out_dir, 'model.log')

    # Load the model
    _model = importlib.import_module(params.model)
    lookahead = params.model_params['L']
    chunk_size = params.model_params['L'] * params.model_params['dec_chunk_size']
    class Net(_model.Net):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @torch.no_grad()
        def forward(self, mixture, label, *bufs):
            return self.predict(mixture, label, *bufs, True)
    model = Net(**params.model_params)

    utils.set_logger(log_file)

    # Load model from checkpoint
    model.eval()
    model.exporting = True
    logging.info(f"Exporting {params.model}...")

    eg_mixed = torch.randn(1, 2, chunk_size + 2 * lookahead)
    eg_label = torch.randn(1, params.model_params['label_len'])
    eg_buffers = model.init_buffers(1, 'cpu')

    logging.info(f"Input shape: {eg_mixed.shape}")
    logging.info(f"Label shape: {eg_label.shape}")
    logging.info(f"Buffers shape: {[_.shape for _ in eg_buffers]}")

    # Export the model
    torch.onnx.export(model,
                      (eg_mixed, eg_label, *eg_buffers),
                      onnx_file,
                      export_params=True,
                      input_names = ['x',
                                     'label',
                                     'init_enc_buf_l',
                                     'init_enc_buf_r',
                                     'init_dec_buf_l',
                                     'init_dec_buf_r',
                                     'init_out_buf_l',
                                     'init_out_buf_r',
                                    ],
                      output_names = ['filtered',
                                      'enc_buf_r',
                                      'enc_buf_l',
                                      'dec_buf_l',
                                      'dec_buf_r',
                                      'out_buf_l',
                                      'out_buf_r',
                                     ],
                      opset_version=OPSET
    )
    logging.info(f"Exported .onnx to {onnx_file}")

    # Check usability and save ORT
    os.system(f"python -m onnxruntime.tools.check_onnx_model_mobile_usability {onnx_file}")
    os.system(f"python -m onnxruntime.tools.convert_onnx_models_to_ort {out_dir}")
    logging.info(f"Exported .ort to {out_dir}")

def save_onnx_conv_tasnet(out_dir, test=False):
    """ Converts a model to onnx file """
    # Load model and training params
    params = utils.Params('experiments/cached_conv_tasnet_binaural_parallel/config.json')
    onnx_file = os.path.join(out_dir, 'model.onnx')
    log_file = os.path.join(out_dir, 'model.log')

    # Load the model
    _model = importlib.import_module(params.model)
    lookahead = params.model_params['L']
    chunk_size = params.model_params['L'] * 13
    class Net(_model.Net):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @torch.no_grad()
        def forward(self, mixture, label, *bufs):
            return self.predict(mixture, label, *bufs, True)
    model = Net(**params.model_params)

    utils.set_logger(log_file)

    # Load model from checkpoint
    model.eval()
    model.exporting = True
    logging.info(f"Exporting {params.model}...")

    eg_mixed = torch.randn(1, 2, chunk_size + 2 * lookahead)
    eg_label = torch.randn(1, params.model_params['label_len'])
    eg_buffers = model.init_buffers(1, 'cpu')

    logging.info(f"Input shape: {eg_mixed.shape}")
    logging.info(f"Label shape: {eg_label.shape}")
    logging.info(f"Buffers shape: {[_.shape for _ in eg_buffers]}")

    # Export the model
    torch.onnx.export(model,
                      (eg_mixed, eg_label, *eg_buffers),
                      onnx_file,
                      export_params=True,
                      input_names = ['x',
                                     'label',
                                     'init_ctx_buf_l',
                                     'init_ctx_buf_r',
                                     'init_out_buf_l',
                                     'init_out_buf_r',
                                    ],
                      output_names = ['filtered',
                                      'out_buf_l',
                                      'out_buf_r',
                                     ],
                      opset_version=OPSET
    )
    logging.info(f"Exported .onnx to {onnx_file}")

    # Check usability and save ORT
    os.system(f"python -m onnxruntime.tools.check_onnx_model_mobile_usability {onnx_file}")
    os.system(f"python -m onnxruntime.tools.convert_onnx_models_to_ort {out_dir}")
    logging.info(f"Exported .ort to {out_dir}")


def save_onnx_single_head(out_dir, test=False):
    """ Converts a model to onnx file """
    # Load model and training params
    params = utils.Params('experiments/sc_waveformer/config.json')
    onnx_file = os.path.join(out_dir, 'model.onnx')
    log_file = os.path.join(out_dir, 'model.log')

    # Load the model
    _model = importlib.import_module(params.model)
    lookahead = params.model_params['L']
    chunk_size = params.model_params['L'] * params.model_params['dec_chunk_size']
    class Net(_model.Net):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @torch.no_grad()
        def forward(self, mixture, label, *bufs):
            return self.predict(mixture, label, *bufs)
    model = Net(**params.model_params)

    utils.set_logger(log_file)

    # Load model from checkpoint
    model.eval()
    model.exporting = True
    logging.info(f"Exporting {params.model}...")

    eg_mixed = torch.randn(1, 2, chunk_size + 2 * lookahead)
    eg_label = torch.randn(1, params.model_params['label_len'])
    eg_buffers = model.init_buffers(1, 'cpu')

    logging.info(f"Input shape: {eg_mixed.shape}")
    logging.info(f"Label shape: {eg_label.shape}")
    logging.info(f"Buffers shape: {[_.shape for _ in eg_buffers]}")

    # Export the model
    torch.onnx.export(model,
                      (eg_mixed, eg_label, *eg_buffers),
                      onnx_file,
                      export_params=True,
                      input_names = ['x',
                                     'label',
                                     'init_enc_buf',
                                     'init_dec_buf',
                                     'init_out_buf'],
                      output_names = ['filtered',
                                      'enc_buf',
                                      'dec_buf',
                                      'out_buf'],
                      opset_version=OPSET
    )
    logging.info(f"Exported .onnx to {onnx_file}")

    # Save IO names and shapes to json
    io = {
        'inputs': [
            ('x', list(eg_mixed.shape)),
            ('label', list(eg_label.shape)),
            ('init_enc_buf', list(eg_buffers[0].shape)),
            ('init_dec_buf', list(eg_buffers[1].shape)),
            ('init_out_buf', list(eg_buffers[2].shape)),
        ],
        'outputs': [
            'filtered',
            'enc_buf',
            'dec_buf',
            'out_buf',
        ]
    }
    with open(os.path.join(out_dir, 'model.io.json'), 'w') as f:
        json.dump(io, f, indent=4)

    # Check usability and save ORT
    os.system(f"python -m onnxruntime.tools.check_onnx_model_mobile_usability {onnx_file}")
    os.system(f"python -m onnxruntime.tools.convert_onnx_models_to_ort {out_dir}")
    logging.info(f"Exported .ort to {out_dir}")

    if test:
        # Load ONNX model
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
        ort_sess = ort.InferenceSession(onnx_file)

        warmup_iters = 20
        n_iters = 100

        test_x = torch.rand(
            1, 2,
            (warmup_iters + n_iters) * chunk_size + 2 * lookahead)
        test_label = torch.zeros(1, params.model_params['label_len'])
        test_label[0, 5] = 1.
        test_enc_buf, test_dec_buf, test_out_buf = model.init_buffers(1, 'cpu')

        # Onnx inference
        x = test_x.numpy()
        label = test_label.numpy()
        enc_buf, dec_buf, out_buf = \
            [_.numpy() for _ in [test_enc_buf, test_dec_buf, test_out_buf]]
        output = torch.zeros_like(test_x[:, :, lookahead:-lookahead]).numpy()

        for i in range(warmup_iters + n_iters):
            if i == warmup_iters:
                start_time = time.time()

            s = chunk_size * i
            e = s + chunk_size
            inputs = {'x': x[:, :, s:e + 2*lookahead],
                      'label': label,
                      'init_enc_buf': enc_buf,
                      'init_dec_buf': dec_buf,
                      'init_out_buf': out_buf}
            chunk_out, enc_buf, dec_buf, out_buf = ort_sess.run(None, inputs)
            output[:, :, s:e] = chunk_out
        end_time = time.time()

        time_elapsed_ms = (end_time - start_time) * 1000
        logging.info("Time per iteration = %.04f ms" % (time_elapsed_ms / n_iters))

        output = torch.from_numpy(output)

        # Pytorch inference
        with torch.no_grad():
            gt_output = model(test_x, test_label, test_enc_buf, test_dec_buf, test_out_buf)
            gt_output = gt_output[0]

        log_allclose(gt_output, output)

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='src/inference/iOSRuntime/SemaudioORT')
    parser.add_argument('--test', action='store_true',
                        help="Whether to test the model for exporting it.")
    args = parser.parse_args()

    save_onnx_conv_tasnet(args.out_dir, args.test)
