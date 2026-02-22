from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import InferenceBundle, build_multimodal_inputs, load_inference_bundle
from clincot.methods.visual_cot import parse_bbox_from_text


@dataclass
class CoTOutput:
    reasoning: str
    bbox: list[float] | None
    answer: str | None = None


class CoTInferencer:
    def __init__(self, config_path: str | Path, checkpoint_path: str | Path) -> None:
        self.bundle = load_inference_bundle(config_path, checkpoint_path)

    def _decode(self, output_ids) -> str:
        return self.bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def infer_bbox(
        self,
        image_path: str | Path,
        question: str,
        max_new_tokens: int = 128,
    ) -> CoTOutput:
        prompt = (
            f"Question: {question}\n"
            "Please provide the bounding box coordinate of the key region."
        )
        inputs = build_multimodal_inputs(self.bundle, image_path=image_path, prompt=prompt)
        output_ids = self.bundle.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = self._decode(output_ids)
        bbox = parse_bbox_from_text(text)
        return CoTOutput(reasoning=text, bbox=bbox)

    def infer_answer_with_bbox(
        self,
        image_path: str | Path,
        question: str,
        bbox_text_or_reasoning: str,
        max_new_tokens: int = 256,
    ) -> CoTOutput:
        prompt = (
            f"Question: {question}\n"
            f"Region hint: {bbox_text_or_reasoning}\n"
            "Please answer based on the original image and local detail image."
        )
        inputs = build_multimodal_inputs(self.bundle, image_path=image_path, prompt=prompt)
        output_ids = self.bundle.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        answer = self._decode(output_ids)
        bbox = parse_bbox_from_text(bbox_text_or_reasoning)
        return CoTOutput(reasoning=bbox_text_or_reasoning, bbox=bbox, answer=answer)
