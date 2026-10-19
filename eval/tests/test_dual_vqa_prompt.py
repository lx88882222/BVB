"""Judge request regressions; no video decoding or paid API calls required."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dual_vqa_metric as metric


class QuestionOptionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.qa = {
            "id": 1, "scene_name": "scene", "dataset": "scannet",
            "question_type": "route_planning",
            "question": "Which sequence of turns reaches the toilet?",
            "options": ["A. Turn Back, Turn Left", "B. Turn Left, Turn Left",
                        "C. Turn Left, Turn Right", "D. Turn Right, Turn Right"],
            "ground_truth": "C",
            "annotation_private": "DO_NOT_SEND_PRIVATE_METADATA",
        }
        self.client = Mock()
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="C"))],
            usage=None,
        )
        video = self.root / "videos/scannet/scene.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
        render = self.root / "run/camera_renders/scene.mp4"
        render.parent.mkdir(parents=True)
        render.touch()

    def request(self, mode, qa):
        metadata = {"1": qa}
        common = dict(client=self.client, metadata=metadata, model="mock-judge", workers=1)
        with patch.object(metric, "sample_frames_b64", return_value=["frame-one", "frame-two"]):
            if mode == "original":
                metric.ensure_originals(**common, vsi_bench=self.root / "videos",
                                        results_dir=self.root / "originals", n_frames=16, qa_ids=None)
            elif mode == "text_only":
                metric.ensure_text_only(**common, results_dir=self.root / "text-only")
            else:
                metric.score_run(**common, run_dir=self.root / "run",
                                 original_bank={"1": {"correct": True}},
                                 n_frames=16, resume=False, force=False)
        return self.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]

    def test_options_reach_all_three_judge_paths_without_answer_leakage(self):
        for mode in ("original", "render", "text_only"):
            with self.subTest(mode=mode):
                content = self.request(mode, self.qa)
                text = content[0]["text"]
                expected = self.qa["question"] + "\n\nOptions:\n" + "\n".join(self.qa["options"])
                self.assertIn(expected, text)
                self.assertEqual(text.count("Options:"), 1)
                self.assertNotIn("ground_truth", text)
                self.assertNotIn("DO_NOT_SEND_PRIVATE_METADATA", text)
                self.assertEqual([part["type"] for part in content],
                                 ["text"] if mode == "text_only" else ["text", "image_url", "image_url"])

    def test_numeric_question_without_options_is_unchanged(self):
        for option_state in ("missing", None, []):
            for mode in ("original", "render", "text_only"):
                with self.subTest(options=option_state, mode=mode):
                    qa = dict(self.qa, question_type="object_size_estimation",
                              question="How long is the bathtub in centimeters?", ground_truth="140")
                    qa.pop("options")
                    if option_state != "missing":
                        qa["options"] = option_state
                    # Original and text-only paths resume by ID; use a new cache per request.
                    with patch.object(metric, "load_answer_bank", return_value={}):
                        content = self.request(mode, qa)
                    text = content[0]["text"]
                    self.assertTrue(text.endswith("Question: " + qa["question"]))
                    self.assertNotIn("Options:", text)
                    self.assertNotIn("140", text)


if __name__ == "__main__":
    unittest.main()
