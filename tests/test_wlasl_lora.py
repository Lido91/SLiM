"""Protocol checks for adapters, exact evaluation, and micro/macro accuracy."""
import copy
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

# Allows tests to exercise the staged module before installing into the checkout.
source = Path(__file__).resolve().parents[1] / "slim/eval/wlasl_lora.py"
if str(source.parents[2]) not in sys.path:
    sys.path.insert(0, str(source.parents[2]))
spec = importlib.util.spec_from_file_location("wlasl_lora_under_test", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LoRAProtocolTest(unittest.TestCase):
    def test_adapter_preserves_initial_outputs_and_only_adapter_learns(self):
        torch.manual_seed(0)
        base = nn.Linear(6, 4)
        x = torch.randn(3, 6)
        expected = base(x).detach()
        original = copy.deepcopy(base.state_dict())
        layer = module.LoRALinear(base)
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=0.01)
        layer(x).square().mean().backward()
        opt.step()
        self.assertGreater(layer.b.abs().sum().item(), 0)
        for name, value in layer.base.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertIsNone(layer.base.weight.grad)

    def test_all_nested_encoder_linears_receive_adapters(self):
        encoder = nn.Sequential(nn.Linear(3, 4), nn.Sequential(nn.ReLU(), nn.Linear(4, 5)))
        names = module.add_lora(encoder)
        self.assertEqual(names, ["0", "1.1"])
        self.assertIsInstance(encoder[1][1], module.LoRALinear)
        self.assertFalse(encoder[0].base.weight.requires_grad)

    def test_exact_sampler_even_when_world_exceeds_length(self):
        for size in (0, 1, 3, 11, 3916, 2878):
            for world in (1, 2, 4, 7):
                shards = [module.ExactEvalSampler(range(size), rank, world) for rank in range(world)]
                indices = [index for shard in shards for index in shard]
                self.assertEqual(sorted(indices), list(range(size)))
                for shard in shards:
                    self.assertEqual(len(shard), len(list(shard)))

    def test_macro_uses_present_classes_and_is_not_micro(self):
        counts = torch.tensor([[3, 3, 3, 3], [1, 0, 1, 1], [0, 0, 0, 0]])
        metrics = module.metrics_from_counts(counts)
        self.assertEqual(metrics["pi_top1"], 75)
        self.assertEqual(metrics["pc_top1"], 50)
        self.assertEqual(metrics["classes_present"], 2)
        self.assertEqual(metrics["pi_top5"], 100)

    def test_test_as_dev_runs_three_trainers_and_does_not_run_a_final_test(self):
        driver_source = Path(__file__).resolve().parents[1] / "scripts/eval_wlasl_lora_compare.py"
        driver_spec = importlib.util.spec_from_file_location("compare_under_test", driver_source)
        driver = importlib.util.module_from_spec(driver_spec)
        driver_spec.loader.exec_module(driver)
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "out"
            for step in driver.STEPS:
                checkpoint = root / "pretrain/eval" / f"training_{step - 1}/teacher_checkpoint.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"synthetic checkpoint placeholder")

            def fake_run(command, **kwargs):
                step = int(command[command.index("--pretrain-step") + 1])
                self.assertEqual(command[command.index("--mode") + 1], "train")
                self.assertEqual(command[command.index("--dev-split") + 1], "test")
                calls.append((step, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                run_dir = Path(command[command.index("--output-dir") + 1])
                metrics = {key: step / 10000 for key in ("pi_top1", "pc_top1", "pi_top5", "pc_top5")}
                (run_dir / "dev_summary.json").write_text(json.dumps({
                    "pretrain_step": step, "best_epoch": 1, "epochs_completed": 1,
                    "best_dev": metrics, "selection_split": "test", "independent_test": False}))

            argv = ["compare", "--output-dir", str(out), "--pretrain-dir", str(root / "pretrain"),
                    "--dev-split", "test", "--wandb-mode", "disabled"]
            with patch.object(driver, "ROOT", root), patch.object(sys, "argv", argv), \
                    patch.object(driver.subprocess, "run", side_effect=fake_run), \
                    contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            self.assertEqual(sorted(calls), [(62500, "0"), (75000, "1"), (87500, "2")])
            result = json.loads((out / "results.json").read_text())
            self.assertFalse(result["independent_test"])
            self.assertEqual(result["selection_split"], "test")
            self.assertEqual(result["selected_pretrain_step"], 87500)
            self.assertNotIn("final_test", result)


if __name__ == "__main__":
    unittest.main()
