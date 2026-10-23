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
uses the language-model steps. `--ocr` recognises PDF pages that have their own text,
`--keras` uses the original TensorFlow models, `--no-fixes` and `--join-lines` change the
corrections (below).

What differs from the original pipeline:

- **PDF text.** A PDF page that carries its own text (a CAD export) is read from that text,
  exactly, symbols included; only image pages are recognised (`edocr2/textlayer.py`).
- **PyTorch, on the GPU, without TensorFlow.** The CRAFT detector (`keras_ocr/craft.py`) and
  both recognizers (`edocr2/torch_recognizer.py`) run in PyTorch with the original weights;
  their outputs match TensorFlow's to rounding (`tests/test_torch_recognizer.py`). Under
  TensorFlow 2.21 on Windows the detector took about two minutes a patch on the CPU; now a page
  takes about 10 s on a laptop RTX 5060. TensorFlow is only needed for `--keras` and to convert
  the released weights once (`setup` does).
- **Fine-tuned recognizers** (`edocr2/models/finetuned`, used when present): trained on crops
  the pipeline cuts from synthetic drawings of public parts, made with drawgen, plus synthetic
  strips in the original fonts (`training/`). The dimension recognizer's alphabet adds the rest
  of the uppercase letters, the lowercase shaft-fit letters, `↧ ⌴ ⌵ □ ×`, brackets and `' % # *`,
  so hole notes (`4X M8-6H THRU`, `↧ 5 MIN, 6 MAX`, `⌴ ⌀11`) are read whole.
- **Corrections** (`edocr2/postfix.py`), each kept for what it measured: a GD&T value read
  without its separator gets it back, `Z` before a number is `⌀`, `B` beside digits is `6`.
- **Robustness.** Crops that crashed the pipeline on scans (empty, blank, one row, no frame
  found) are handled (`tests/test_robustness.py`); text files are read as UTF-8 on Windows.

### Measured

`benchmarks/paper.py` is the paper's benchmark (the labelled drawings in `tests/test_samples`,
scored as `test_all.py` scores them; that script no longer runs). `benchmarks/drawgen.py`
scores every callout on 16 drawgen sheets of eight public NIST parts that no training data
came from, clean and "scanned" (rotated, blurred, noisy, one-bit), against each sheet's
record. `benchmarks/ablate.py` measures each correction alone. Results are in
`benchmarks/results/`.

| | Original pipeline | This version |
|---|---|---|
| Paper benchmark: recall / precision / CER | 93.75% / 96.2% / 0.73% | 93.75% / 94.9% / **0.24%** |
| GD&T frames fully right (value, symbol, datums), clean / scanned | 43% / 47% | **100% / 86%** |
| Thread notes fully right, clean / scanned | 59% / 56% | **97% / 84%** |
| Sizes fully right, clean / scanned | 60% / 60% | **90% / 100%** |
| Ra, clean / scanned | 88% / 81% | **100% / 94%** |
| Seconds a page (A3, RTX 5060) | minutes (CPU, TensorFlow 2.21) | about 10 |

"Original pipeline" is this repository's base with the PyTorch detector (identical output) and
the released recognizers. The drawgen sheets are synthetic; their style follows NX drawings.
Not yet good: free-text notes (38-50% read exactly; they go to Tesseract), stacked deviations in
brackets such as `(+0.025/0)`, and hole notes on sizes (6 of 9 fully right, a small sample).

### Training

```
benchmarks\make_drawgen_sets.ps1 -Drawgen <drawgen> -Out <data>      # public parts; eval parts kept apart
python training/collect_crops.py <data>/drawgen/train --out <data>/crops
python training/pseudo_label.py tests/*.PDF --out <data>/pseudo        # keeps the original style
python training/train.py --kind gdt --crops <data>/crops --init edocr2/models/recognizer_gdts.pt --out recognizer_gdts.pt --epochs 10
python training/train.py --kind dim --crops <data>/crops <data>/pseudo --init edocr2/models/recognizer_dimensions_2.pt --out recognizer_dimensions_2.pt --epochs 25 --synth 1.0
```

About an hour of crop collection and 25 minutes of training on a laptop GPU.

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