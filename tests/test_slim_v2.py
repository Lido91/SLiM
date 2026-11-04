"""V2 contracts: retained detail, masking, validity, loss, EMA and checkpoint portability."""
import copy
import tempfile
from pathlib import Path
import unittest
import itertools

import numpy as np
import torch
from omegaconf import OmegaConf

from slim.configs import slim_default_config
from slim.data.augmentations import DataAugmentationDINO
from slim.data.collate import collate_data_and_cast
from slim.data.masking import MaskingGenerator
from slim.data.skeletons import get_skeleton
from slim.hub import load_encoder_from_pth
from slim.models import build_model_from_cfg
from slim.models.slim_v2 import SlimV2Encoder
from slim.train.ssl_meta_arch import SSLMetaArch
from slim.train.v2_objective import balanced_assignments, parent_targets
from slim.eval.wlasl_lora import SignLoRAClassifier
from tools.export_encoder import export
from slim.data.samplers import ShardedInfiniteSampler
from slim.train.v2_state import V2TrainState


def clips(batch=2, frames=64):
    x = torch.randn(batch, 3, frames, 69, 1) * 0.1
    x[:, 2] = 1
    return x


def cfg():
    root = Path(__file__).resolve().parents[1]
    c = OmegaConf.merge(OmegaConf.create(slim_default_config),
                        OmegaConf.load(root / "slim/configs/pretrain/slim_sign_v2_openasl.yaml"))
    c.student.v2.update(embed_dim=32, depth=2, num_heads=4)
    c.dino.update(head_n_prototypes=16, head_hidden_dim=32, head_bottleneck_dim=16)
    c.ibot.update(head_n_prototypes=24, head_hidden_dim=32, head_bottleneck_dim=16)
    c.compute_precision.grad_scaler = False
    return c


def batch(all_missing=False):
    spec = get_skeleton("sign69")
    aug = DataAugmentationDINO(8, skeleton=spec, max_window=128,
                              parent_matching=True, min_global_overlap=0.5)
    raw = clips(2, 128).numpy()
    if all_missing:
        raw[:] = 0
    samples = [(aug(a), ()) for a in raw]
    kw = dict(joint_groups=spec.mask_groups, group_weights=[len(g)**0.5 for g in spec.mask_groups])
    gen = MaskingGenerator((16, 69), min_num_patches=2, max_num_patches=552, **kw)
    locals_ = {f"local_crops_{n}": MaskingGenerator((n // 4, 69), min_num_patches=1,
                                                   max_num_patches=n // 4 * 69, **kw) for n in (8, 16, 32)}
    return collate_data_and_cast(samples, (0.5, 0.8), (0.5, 0.8), False, 1.0,
                                 torch.float32, n_tokens=1104, mask_generator=gen,
                                 local_mask_generator=locals_, respect_validity=True,
                                 max_mfp_tokens_per_view=32)


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(7)
        return SlimV2Encoder(embed_dim=32, depth=2, num_heads=4).eval()

    def test_global_tokens_compact_joint_detail_retained(self):
        out = self.model().forward_features(clips())
        self.assertEqual(out["x_norm_parttokens"].shape, (2, 448, 32))
        self.assertEqual(out["x_norm_patchtokens"].shape, (2, 1104, 32))
        self.assertEqual(out["x_norm_embedding"].shape, (2, 32))

    def test_missing_coordinates_do_not_change_any_features(self):
        model, x = self.model(), clips()
        x[:, 2, :, 9:30] = 0
        y = x.clone()
        y[:, :2, :, 9:30] = 1e6
        a, b = model.forward_features(x), model.forward_features(y)
        for key in ("x_norm_clstoken", "x_norm_embedding", "x_norm_patchtokens"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(a["x_norm_patchtokens"].reshape(2, 16, 69, 32)[:, :, 9:30].abs().sum(), 0)

    def test_artificial_mask_precedes_every_joint_mixing_operation(self):
        model, x = self.model(), clips()
        mask = torch.zeros(2, 16, 69, dtype=torch.bool)
        mask[:, 3:8, 10:25] = True
        y = x.clone()
        y[:, :2, 12:32, 10:25] += 100
        a, b = model.forward_features(x, mask.flatten(1)), model.forward_features(y, mask.flatten(1))
        for key in ("x_norm_clstoken", "x_norm_patchtokens"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_padding_does_not_change_valid_outputs_or_embedding(self):
        model, x = self.model(), clips(frames=32)
        padded = torch.cat((x, torch.zeros(2, 3, 16, 69, 1)), 2)
        a, b = model.forward_features(x), model.forward_features(padded)
        for key in ("x_norm_clstoken", "x_norm_embedding"):
            torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(a["x_norm_patchtokens"], b["x_norm_patchtokens"][:, :552], rtol=1e-5, atol=2e-6)

    def test_absent_and_nonfinite_inputs_are_finite(self):
        x = clips()
        x[:, 2] = 0
        x[:, 0] = float('nan')
        out = self.model().forward_features(x)
        self.assertTrue(all(torch.isfinite(out[k]).all() for k in ("x_norm_clstoken", "x_norm_patchtokens", "x_norm_embedding")))
        self.assertEqual(out["x_norm_patchtokens"].abs().sum(), 0)
        self.assertEqual(out["x_norm_embedding"].abs().sum(), 0)

    def test_parent_targets_preserve_crop_major_sample_identity(self):
        targets = torch.arange(4)[:, None]
        parents = torch.tensor([0, 0, 1, 1, 0, 0])
        self.assertEqual(parent_targets(targets, parents, 2).flatten().tolist(), [0, 1, 2, 3, 0, 1])

    def test_collate_validity_parent_metadata_and_bounded_joint_targets(self):
        x = batch()
        self.assertTrue((x["mfp_masks"] <= x["collated_masks"]).all())
        self.assertTrue((x["collated_masks"] <= x["global_patch_valid"]).all())
        self.assertTrue((x["mfp_masks"].sum(1) <= 32).all())
        self.assertEqual(x["local_parents_32"].tolist(), [0, 0, 1, 1])
        self.assertEqual(x["local_masks_8"].shape, (4, 138))
        self.assertFalse(x["local_masks_8"].any())

    def test_sinkhorn_finite_normalized_and_empty(self):
        for logits in (torch.randn(32, 16), torch.randn(32, 16) * 1e4):
            p = balanced_assignments(logits, 0.04)
            self.assertTrue(torch.isfinite(p).all())
            torch.testing.assert_close(p.sum(1), torch.ones(32), rtol=1e-5, atol=1e-5)
        self.assertEqual(balanced_assignments(torch.empty(0, 16), 0.04).shape, (0, 16))

    def test_complete_ssl_backward_teacher_stopgrad_and_ema(self):
        model = SSLMetaArch(cfg())
        torch.testing.assert_close(next(model.student.parameters()), next(model.teacher.parameters()))
        before = copy.deepcopy(model.teacher.state_dict())
        losses = model.forward_backward(batch(), teacher_temp=0.04)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        for name in ("patch_embed.proj.weight", "part_queries", "blocks.0.attn.qkv.weight", "context_proj.weight"):
            grad = dict(model.student.backbone.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum(), 0, name)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        opt = torch.optim.AdamW(model.student.parameters(), lr=1e-3)
        opt.step()
        model.update_teacher(0.9)
        for key, value in model.teacher.state_dict().items():
            torch.testing.assert_close(value, before[key] * 0.9 + model.student.state_dict()[key] * 0.1)

    def test_all_missing_batch_is_zero_loss_and_backward_safe(self):
        model = SSLMetaArch(cfg())
        losses = model.forward_backward(batch(all_missing=True), teacher_temp=0.04)
        self.assertEqual(sum(v.item() for v in losses.values()), 0)

    def test_checkpoint_export_load_and_lora_classification(self):
        model, x = self.model(), clips()
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / 'teacher.pth', Path(temp) / 'encoder.pth'
            torch.save({'teacher': {'backbone.'+k:v for k,v in model.state_dict().items()},
                        'encoder_config': model.get_config()}, src)
            export(str(src), str(dst))
            loaded = load_encoder_from_pth(dst).eval()
            torch.testing.assert_close(model(x), loaded(x), rtol=0, atol=0)
            classifier = SignLoRAClassifier(loaded, classes=5).eval()
            self.assertEqual(classifier(x).shape, (2, 5))
            classifier(x).square().sum().backward()
            self.assertTrue(any(p.grad is not None for n,p in classifier.encoder.named_parameters() if n.endswith('.b')))

    def test_build_factory_and_intermediate_layers(self):
        student, teacher, dim = build_model_from_cfg(cfg())
        self.assertEqual(dim, 32)
        outputs = student.get_intermediate_layers(clips(), 2, return_class_token=True)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0][0].shape, (2, 1104, 32))

    def test_resume_sampler_matches_uninterrupted_across_epochs_and_ranks(self):
        for world in (1, 2, 3):
            for rank in range(world):
                args = dict(sample_count=11, seed=7, shuffle=True, start=rank, step=world)
                full = list(itertools.islice(ShardedInfiniteSampler(**args), 50))
                for advance in (1, 12, 27):
                    resumed = list(itertools.islice(ShardedInfiniteSampler(**args, advance=advance), 10))
                    self.assertEqual(full[advance:advance+10], resumed)

    def test_resume_guard_rejects_changed_architecture_and_restores_rng(self):
        c = cfg()
        state = V2TrainState(c)
        saved = state.state_dict()
        expected = torch.rand(3)
        state.load_state_dict(saved)
        torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)
        c.student.v2.part_tokens = [4, 4, 4, 4]
        with self.assertRaisesRegex(ValueError, 'configuration/data changed'):
            V2TrainState(c).load_state_dict(saved)


if __name__ == '__main__':
    unittest.main()
