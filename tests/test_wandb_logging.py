import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from slim.logging import helpers
from slim.logging import wandb as tracking


class LoggingTests(unittest.TestCase):
    def cfg(self, root, enabled=True):
        return OmegaConf.create({'train': {'output_dir': root}, 'wandb': {
            'enabled': enabled, 'project': 'SLiM', 'entity': None, 'name': None,
            'id': None, 'mode': 'online'}})

    def sdk(self):
        run = MagicMock()
        run.id = 'unit-run'
        run.project = 'SLiM'
        run.entity = 'test-entity'
        run.url = 'https://wandb.ai/test-entity/SLiM/runs/unit-run'
        sdk = types.SimpleNamespace(init=MagicMock(return_value=run),
            Settings=MagicMock(), util=types.SimpleNamespace(generate_id=MagicMock(return_value='unit-run')))
        return sdk, run

    def test_disabled(self):
        with tracking.track_training(self.cfg('/unused', False)) as sink:
            self.assertIsNone(sink)

    def test_worker_does_not_create_run(self):
        sdk, _ = self.sdk()
        with patch.object(tracking.distributed, 'is_main_process', return_value=False), patch.dict(sys.modules, {'wandb': sdk}):
            with tracking.track_training(self.cfg('/unused')) as sink:
                self.assertIsNone(sink)
        sdk.init.assert_not_called()

    def test_metrics_equal_json_and_resume_id(self):
        sdk, run = self.sdk()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'wandb': sdk}), patch.object(tracking.distributed, 'is_main_process', return_value=True), patch.object(helpers.distributed, 'is_main_process', return_value=True), patch.object(helpers.torch.cuda, 'is_available', return_value=False):
            with tracking.track_training(self.cfg(tmp)) as sink:
                log = helpers.MetricLogger(output_file=str(Path(tmp) / 'metrics.json'), metric_sink=sink)
                log.update(total_loss=12.5, global_batch_size=256)
                log.dump_in_output_file(10, 1.1, .01)
            saved = json.loads((Path(tmp) / 'metrics.json').read_text())
            run.log.assert_called_once_with(saved, step=10)
            self.assertEqual(saved['global_batch_size'], 256)
            self.assertEqual(json.loads((Path(tmp) / 'wandb_run.json').read_text())['id'], 'unit-run')
            with tracking.track_training(self.cfg(tmp)):
                pass
            self.assertEqual(sdk.init.call_args.kwargs['id'], 'unit-run')
            sdk.util.generate_id.assert_called_once()
            run.finish.assert_called_with(exit_code=0)

    def test_exception_finishes_failed_run(self):
        sdk, run = self.sdk()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'wandb': sdk}), patch.object(tracking.distributed, 'is_main_process', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                with tracking.track_training(self.cfg(tmp)):
                    raise RuntimeError('synthetic failure')
        run.finish.assert_called_once_with(exit_code=1)

    def test_worker_does_not_write_metrics(self):
        sink = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, patch.object(helpers.distributed, 'is_main_process', return_value=False):
            out = Path(tmp) / 'metrics.json'
            log = helpers.MetricLogger(output_file=str(out), metric_sink=sink)
            log.dump_in_output_file(0, 0., 0.)
            self.assertFalse(out.exists())
            sink.assert_not_called()

    def test_callback_without_json_file(self):
        sink = MagicMock()
        with patch.object(helpers.distributed, 'is_main_process', return_value=True), patch.object(helpers.torch.cuda, 'is_available', return_value=False):
            log = helpers.MetricLogger(metric_sink=sink)
            log.update(total_loss=1.)
            log.dump_in_output_file(0, .1, .01)
        self.assertEqual(sink.call_args.args[0]['total_loss'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
