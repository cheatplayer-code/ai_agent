"""Inference harness for the baseline LLM.

This module provides utilities to run baseline inference over a JSONL
dataset.  It supports a mock backend for testing and a transformers
backend for local model inference.  The harness reads input rows,
builds prompts, invokes the chosen backend, parses and validates
responses, and writes out a rich JSONL file with both raw and
structured information for later evaluation.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .prompting import build_prompt
from .parser import parse_model_response

logger = logging.getLogger(__name__)


class LLMBackend:
    """Base class for model backends.

    Subclasses must implement a `generate` method that takes a prompt
    string and returns a model response string.
    """

    name: str

    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class MockBackend(LLMBackend):
    """A simple mock backend that returns a canned response.

    Useful for tests and smoke runs when no real model is available.  It
    constructs a trivial JSON string using placeholder text and an
    empty recommendation list.  Users may extend this class to
    customise mock behaviour.
    """

    def __init__(self) -> None:
        self.name = "mock"

    def generate(self, prompt: str) -> str:
        # For determinism, derive a simple summary from the first line of the
        # source.  If the report is multiline, take the first non-empty line.
        # This is purely heuristic and not intended to produce a useful
        # summary; it suffices for testing the harness.
        source_lines = prompt.split("Report:\n", 1)[-1].split("\n")
        first_line = next((line.strip() for line in source_lines if line.strip()), "")
        summary = (
            first_line[:200] + ("..." if len(first_line) > 200 else "")
            if first_line
            else "Mock summary"
        )
        output = {
            "executive_summary": summary,
            "main_finding": summary,
            "recommendations": [],
            "confidence_reason": "Baseline mock backend used; no confidence can be inferred.",
        }
        return json.dumps(output)


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    class TransformersBackend(LLMBackend):
        """Backend using a Hugging Face causal language model.

        Parameters
        ----------
        model_id: str
            The name or path of the Hugging Face model to load.
        device: str, optional
            Device to run inference on (e.g., 'cpu' or 'cuda').  Defaults
            to CPU if CUDA is not available.
        max_new_tokens: int, optional
            Maximum number of tokens to generate.  Defaults to 256.
        temperature: float, optional
            Softmax temperature.  Lower values make the output more
            deterministic.  Defaults to 0.7.
        top_p: float, optional
            Nucleus sampling parameter.  Defaults to 0.95.
        """

        def __init__(
            self,
            model_id: str,
            device: Optional[str] = None,
            max_new_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.95,
        ) -> None:
            self.name = "transformers"
            self.model_id = model_id
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.top_p = top_p
            # Lazy loading: models are loaded on first use to avoid slow
            # initialisation during test discovery.
            self._model: Optional[AutoModelForCausalLM] = None
            self._tokenizer: Optional[AutoTokenizer] = None

        def _ensure_loaded(self) -> None:
            if self._model is not None and self._tokenizer is not None:
                return
            logger.info("Loading transformers model '%s' on %s", self.model_id, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)

        def generate(self, prompt: str) -> str:
            self._ensure_loaded()
            assert self._model is not None and self._tokenizer is not None
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            # Skip the input prompt to get only the generated part
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            return text

except Exception:  # pragma: no cover
    # transformers not available; define a placeholder backend that errors on use
    class TransformersBackend(LLMBackend):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.name = "transformers"

        def generate(self, prompt: str) -> str:
            raise RuntimeError(
                "Transformers backend is unavailable because the transformers library could not be imported."
            )


def load_backend(
    backend_name: str,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> LLMBackend:
    """Instantiate the desired backend.

    Supported backends:
    - "mock": returns a canned response for testing.
    - "transformers": loads a Hugging Face causal language model.

    Parameters
    ----------
    backend_name: str
        Name of the backend ("mock" or "transformers").
    model_id: Optional[str]
        Hugging Face model identifier (for transformers backend).
    device: Optional[str]
        Device to run inference on (for transformers backend).
    max_new_tokens, temperature, top_p:
        Generation parameters for transformers backend.

    Returns
    -------
    LLMBackend
        An instance of the requested backend.
    """
    name = backend_name.lower()
    if name == "mock":
        return MockBackend()
    if name == "transformers":
        if model_id is None:
            raise ValueError("model_id must be provided for transformers backend")
        return TransformersBackend(
            model_id=model_id,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    raise ValueError(f"Unsupported backend: {backend_name}")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file.

    Parameters
    ----------
    path: Path
        Path to the JSONL file.

    Yields
    ------
    dict
        Parsed JSON objects for each line in the file.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Skipping invalid JSON line: %s", exc)


def run_inference(
    input_path: Path,
    output_path: Path,
    backend_name: str,
    model_id: Optional[str] = None,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    split: Optional[str] = None,
    prompt_variant: str = "strict_json_minimal",
) -> None:
    """Run baseline inference over a JSONL dataset.

    This function reads the input file, constructs prompts from the
    `source` field of each row, generates model outputs using the
    specified backend, parses and validates the responses, and writes
    the results to an output JSONL file.  If `max_samples` is
    provided, only that many rows are processed.

    Parameters
    ----------
    input_path: Path
        Path to the input JSONL file.  Each line must be a JSON object
        containing at least a `source` field.
    output_path: Path
        Path to the output JSONL file.  It will be overwritten if it
        already exists.
    backend_name: str
        Name of the backend to use ("mock" or "transformers").
    model_id: Optional[str]
        Model identifier for the transformers backend.
    max_samples: Optional[int]
        Maximum number of examples to process.  None means process
        all.  Useful for quick smoke tests.
    device, max_new_tokens, temperature, top_p:
        Generation parameters for the transformers backend.
    split: Optional[str]
        A label to attach to each output row (e.g., "train", "valid",
        "test").  If provided, it is stored in the output metadata.
    """
    backend = load_backend(
        backend_name,
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    logger.info(
        "Running inference on %s with backend=%s, max_samples=%s", input_path, backend.name, max_samples
    )

    # Resolve prompt builder based on the variant.  The default variant
    # corresponds to the original behaviour.  A ValueError will be
    # propagated if an unknown variant is requested.
    from .prompt_variants import get_prompt_builder

    prompt_builder = get_prompt_builder(prompt_variant)
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(_iter_jsonl(input_path)):
            if max_samples is not None and count >= max_samples:
                break
            source = row.get("source")
            if source is None:
                logger.warning("Row %s missing 'source'; skipping", idx)
                continue
            # Build a prompt using the selected variant
            prompt = prompt_builder(source)
            try:
                raw_response = backend.generate(prompt)
            except Exception as exc:  # pragma: no cover
                # Capture backend errors as parse failures
                parse_success = False
                parsed_output = None
                parse_error = f"Backend error: {exc}"
            else:
                parse_success, parsed_output, parse_error = parse_model_response(raw_response)
            output_row: Dict[str, Any] = {
                "id": row.get("id", idx),
                "domain": row.get("domain"),
                "family_id": row.get("family_id"),
                "split": split or row.get("split"),
                "raw_response": raw_response if isinstance(raw_response, str) else None,
                "parsed_output": parsed_output,
                "parse_success": parse_success,
                "parse_error": parse_error,
                "backend": backend.name,
                "model": getattr(backend, "model_id", None),
                "prompt_length": len(prompt),
                "prompt_variant": prompt_variant,
            }
            fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            count += 1
