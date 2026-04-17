# PaddleOCR-VL-1.5 pipeline (no Mistral OCR)

**Primary supervised model:** LoRA fine-tuning of [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) on Yorùbá line crops (scripts **14 → 16 → 15** with adapter), on the **same** images and labels as PP-OCRv4. Zero-shot VL-1.5 (**15** without adapter) is a baseline. **Mistral OCR is not implemented** here (use Tesseract, Qwen VL, PP-OCRv4, and the rows above for the comparison table).

## Data safety

- **`data/processed/`** is never overwritten by VL scripts.
- **14** writes only under `data/paddleocr_vl15_sft/` (JSONL + `manifest.json`).
- **15–16** read images by path; ground-truth strings stay NFC-normalised like the rest of the pipeline.

## Steps

1. **Export** (once per consolidated dataset version):

   ```bash
   bash scripts/shell/phase_14_export_vl15.sh
   ```

2. **Zero-shot eval** (GPU; large download):

   ```bash
   export SKIP_VL15_EVAL=0
   bash scripts/shell/phase_15_eval_vl15.sh
   ```

   Installs: `pip install 'transformers>=5' peft accelerate` (and `bitsandbytes` if using `--quantize-4bit`).

3. **LoRA fine-tuning** (long run):

   ```bash
   pip install peft
   bash scripts/shell/phase_16_train_vl15_lora.sh
   ```

4. **Eval fine-tuned adapter**:

   ```bash
   python scripts/15_baseline_paddleocr_vl15.py --adapter-path experiments/paddleocr_vl15_lora/adapter
   ```

5. **Compile tables**: `bash scripts/shell/phase_09_compile.sh`

## Dictionary

Character constraints for **PP-OCRv4 training** live in `data/processed/dictionary/`. The VL model uses a **subword tokenizer**; DER/CER/WER are still computed on **Unicode NFC** strings. Optional dictionary path is recorded in **14**’s `manifest.json` for provenance.

## Training caveat (script 16)

`16_train_paddleocr_vl_lora.py` optimises a **full-sequence** causal loss for reproducibility. For stronger results, mask user tokens or use an external SFT toolkit with the **same** JSONL export — without changing source labels.
