import json
import re
import random
import datetime
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MultipleChoiceEvaluator:
    """
    Evaluates multiple‑choice (A–E) question datasets,
    optionally scoring a Yes/No follow‑up (“ask_followup”: true).
    """

    def __init__(self, dataset_path: str, output_dir: str,
                 delay_between_requests: float = 5.0):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.delay_between_requests = delay_between_requests
        os.makedirs(output_dir, exist_ok=True)

        self.dataset = self._load_dataset()
        random.seed(12345)

    # ---------- Data loading ---------- #
    def _load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} questions")
        return data


    # ---------- Prompt construction ---------- #
    def _build_prompt(self,
                      q: Dict[str, Any]) -> str:
        """
        Build a zero‑shot prompt for a single MCQ.
        """
        prompt = ("Please answer the following multiple‑choice question by "
                  "responding **only** with the single letter (A‑E) of your choice."
                  "\n\n")

        prompt += f"Question: {q['question']}\n"
        for opt in q['options']:
            prompt += f"{opt}\n"

        if q.get("ask_followup", False):
            prompt += ("\nWould you personally make this choice? "
                       "Answer Yes or No, after the letter, separated by a space.\n")

        return prompt

    # ---------- Response parsing ---------- #
    def _extract(self, text: str, expects_follow: bool
                 ) -> Tuple[Optional[str], Optional[str]]:
        if not text:
            return None, None

        choice_match = re.search(r"\b([A-E])\b", text.upper())
        choice = choice_match.group(1) if choice_match else None

        follow = None
        if expects_follow:
            if re.search(r"\byes\b", text, re.IGNORECASE):
                follow = "Yes"
            elif re.search(r"\bno\b", text, re.IGNORECASE):
                follow = "No"

        return choice, follow

    # ---------- Evaluation loop ---------- #
    def evaluate_model(self, model,
                       batch_size: int = 5,
                       batch_delay: int = 60) -> Dict[str, Any]:

        results: List[Dict[str, Any]] = []
        total = len(self.dataset)
        batches = (total + batch_size - 1) // batch_size

        for bi in range(batches):
            chunk = self.dataset[bi*batch_size:(bi+1)*batch_size]

            for q in tqdm(chunk, desc=f"Batch {bi+1}/{batches}"):

                prompt = self._build_prompt(q)
                try:
                    raw = model.generate_response(prompt)
                except Exception as e:
                    logger.error(f"Model error: {e}")
                    raw = ""

                choice, follow = self._extract(raw, q.get("ask_followup", False))

                q_result = {
                    **q,
                    "model_output": raw,
                    "pred": choice,
                    "followup_pred": follow,
                    "is_correct": choice == q['answer']
                }

                if q.get("ask_followup", False) and q.get("followup_correct") is not None:
                    q_result["followup_correctness"] = (
                        follow == q["followup_correct"])
                results.append(q_result)

                time.sleep(self.delay_between_requests)

            if bi < batches-1:
                time.sleep(batch_delay)

        # write results
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.output_dir, f"results_{ts}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved raw results → {out_path}")

        # summary
        correct = sum(r["is_correct"] for r in results)
        acc = correct/len(results)
        summary = {"accuracy": acc, "total": len(results), "correct": correct}
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Overall accuracy {acc:.3f} ({correct}/{len(results)})")
        return summary
