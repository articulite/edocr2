# eDOCr2

A tool for performing segmentation and OCR on engineering drawings, primarily focused on mechanical or production drawings.

## Installation

Detailed installation steps can be found [here](https://github.com/javvi51/edocr2/blob/main/docs/install.md).

## Windows: command line

On Windows, `edocr2.cmd` sets up and runs everything from the repository folder. It needs
[uv](https://docs.astral.sh/uv/) and [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
(`winget install --id UB-Mannheim.TesseractOCR`); no WSL.

```
edocr2.cmd setup                                   # .venv with PyTorch (CUDA when there is an NVIDIA GPU), models, Tesseract languages
edocr2.cmd doctor                                  # what was found
edocr2.cmd run DRAWINGS_FOLDER --out RESULTS       # PDFs (every page, 300 dpi) and image files
```

`run` writes one JSON per page (GD&T frames, dimensions, other text, title-block tables), a
`summary.csv` with one row per page (frames, tightest frame tolerance, toleranced dimensions,
fit classes, threads, finest Ra), and prints totals only. It makes no network calls and never
uses the language-model steps.

What differs from the original pipeline:

- The text detector runs in PyTorch (`TorchDetector`, the CRAFT model with its original
  weights). Its output matches the Keras detector's to rounding; on the CPU it takes under a
  second per patch where the Keras model took about two minutes under TensorFlow 2.21, and
  under 0.1 s on a GPU. `--keras-detector` uses the original.
- The recognizers still run in TensorFlow, on the CPU.
- Crops that used to crash the pipeline on scanned drawings (empty, blank, one-row, no frame
  found) are handled; see `tests/test_robustness.py`.

Measured on 17 synthetic A3 drawings with known callouts, scanned (rotated, blurred, noisy,
1-bit), about 14 s a sheet on a laptop RTX 5060: 57% of GD&T frames recognised as frames
(46% with the right symbol), 81% of thread sizes, 93% of count prefixes like `4X`, 85% of
basic dimensions, 75% of Ra values. Stacked deviations such as `(+0.025/0)` are rarely read.

## How to Use

For quick testing, run the `test_drawing.py` file after downloading the recognizer models from [Releases](https://github.com/javvi51/edocr2/releases).

Other files are provided for additional functionality:
- **`test_train.py`**: For training custom synthetic recognizers or detectors.
- **`test_all.py`**: For benchmarking against all drawings available in the `/tests` folder.
- **`test_llm.py`**: For integration with language models (LLMs).

For more detailed information about using these files, refer to the [Examples](https://github.com/javvi51/edocr2/blob/main/docs/examples.md) documentation.

## Citation

If you find this project useful in your research, please consider citing our paper (preprint):  
[http://dx.doi.org/10.2139/ssrn.5045921](http://dx.doi.org/10.2139/ssrn.5045921)