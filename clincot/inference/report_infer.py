from __future__ import annotations

from pathlib import Path

from .common import build_multimodal_inputs, load_inference_bundle


class ReportInferencer:
    def __init__(self, config_path: str | Path, checkpoint_path: str | Path) -> None:
        self.bundle = load_inference_bundle(config_path, checkpoint_path)

    def infer_one(
        self,
        image_path: str | Path,
        instruction: str = "Generate a concise radiology report.",
        max_new_tokens: int = 256,
    ) -> str:
        prompt = f"Instruction: {instruction}\nReport:"
        inputs = build_multimodal_inputs(self.bundle, image_path=image_path, prompt=prompt)
        output_ids = self.bundle.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def infer_batch(self, samples: list[dict], max_new_tokens: int = 256) -> list[dict]:
        outputs = []
        for row in samples:
            pred = self.infer_one(row["image"], row.get("instruction", "Generate a concise radiology report."), max_new_tokens=max_new_tokens)
            out = dict(row)
            out["pred"] = pred
            outputs.append(out)
        return outputs
