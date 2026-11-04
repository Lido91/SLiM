"""Hierarchical sign-pose encoder: local joints, compact global parts, joint detail.

The 69-joint grid is retained for MFP. Only part summaries enter global attention.
Detection validity and artificial masking are separate throughout the model.
"""
import torch
from torch import nn
from torch.nn import functional as F

from slim.layers.attention import RotaryEmbedding, apply_rotary_pos_emb
from slim.layers.drop_path import DropPath
from slim.layers.swiglu_ffn import SwiGLUFFNFused
from slim.layers.patch_embed import PatchEmbed


PART_SLICES = ((0, 9), (9, 30), (30, 51), (51, 69))


def patch_validity(clips, patch_size=4):
    """[B,3,T,69,1] -> [B,T/P,69]; a patch contains at least one detected frame."""
    if clips.ndim != 5 or clips.shape[1] != 3 or clips.shape[3:] != (69, 1):
        raise ValueError("SLiM V2 expects [B,3,T,69,1] sign69 clips")
    if clips.shape[2] % patch_size:
        raise ValueError("T must be divisible by the temporal patch size")
    present = torch.isfinite(clips).all(1) & (clips[:, 2] > 0)
    return present[..., 0].reshape(clips.shape[0], -1, patch_size, 69).any(2)


class ValidAttention(nn.Module):
    def __init__(self, dim, heads, rope=False, prefix=0):
        super().__init__()
        if dim % heads or (rope and (dim // heads) % 2):
            raise ValueError("attention dimensions must divide evenly; RoPE head dim must be even")
        self.heads, self.prefix = heads, prefix
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(dim // heads) if rope else None

    def forward(self, x, valid, num_t=None, all_valid=False):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).unbind(2)
        if self.rope is not None:
            cos, sin = self.rope(num_t, (n - self.prefix) // num_t, x.device, x.dtype)
            q_data, k_data = apply_rotary_pos_emb(q[:, self.prefix:], k[:, self.prefix:], cos, sin)
            q = torch.cat((q[:, :self.prefix], q_data), 1)
            k = torch.cat((k[:, :self.prefix], k_data), 1)
        # One dummy key avoids all-masked softmax on absent parts; its queries are zeroed below.
        key_mask = None
        if not all_valid:
            safe = valid.clone()
            safe[:, 0] |= ~valid.any(1)
            key_mask = safe[:, None, None]
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2), attn_mask=key_mask)
        y = y.transpose(1, 2).reshape(b, n, d)
        return self.proj(y) * valid[..., None]


class ValidBlock(nn.Module):
    def __init__(self, dim, heads, expansion=4, rope=False, prefix=0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = ValidAttention(dim, heads, rope, prefix)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLUFFNFused(dim, int(expansion * dim))
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x, valid, num_t=None, all_valid=False):
        x = x + self.drop_path(self.attn(self.norm1(x), valid, num_t, all_valid))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x * valid[..., None]



class SlimV2Encoder(nn.Module):
    """69-joint detail grid plus 4/8/8/8 part summaries per temporal patch.

    Local blocks mix joints only within one part and temporal patch. Global blocks
    mix summaries over parts and time. Global context is projected back to each
    joint before the MFP head and validity-aware downstream pooling.
    """
    def __init__(self, img_size=(64, 69), patch_size=(4, 1), in_chans=3,
                 embed_dim=256, depth=8, num_heads=8, local_depth=1,
                 part_tokens=(4, 8, 8, 8), num_register_tokens=4,
                 drop_path_rate=0.0, drop_path_uniform=True):
        super().__init__()
        if tuple(patch_size)[1] != 1 or tuple(img_size)[1] != 69 or in_chans != 3:
            raise ValueError("SLiM V2 requires sign69, three channels and P_J=1")
        if len(part_tokens) != 4 or any(int(k) != k or k < 1 for k in part_tokens):
            raise ValueError("part_tokens must contain four positive integers")
        self.embed_dim = self.num_features = embed_dim
        self.patch_size = tuple(patch_size)
        self.n_blocks, self.num_heads = depth, num_heads
        self.chunked_blocks = False
        self.num_register_tokens = num_register_tokens
        self.part_tokens = tuple(part_tokens)
        self.summaries_per_time = sum(part_tokens)
        self.model_config = dict(arch="slim_v2", img_size=list(img_size), patch_size=list(patch_size),
                                 in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                 local_depth=local_depth, part_tokens=list(part_tokens),
                                 num_register_tokens=num_register_tokens, drop_path_rate=drop_path_rate,
                                 drop_path_uniform=drop_path_uniform)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.joint_pos = nn.Parameter(torch.zeros(1, 1, 69, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # Process all four padded parts together. Shared local weights avoid four
        # sequences of small GPU kernels; joint positions preserve part identity.
        self.local_encoders = nn.ModuleList([ValidBlock(embed_dim, num_heads, expansion=2)
                                             for _ in range(local_depth)])
        self.part_queries = nn.Parameter(torch.empty(4, max(part_tokens), embed_dim))
        self.pool_norm = nn.LayerNorm(embed_dim)
        self.pool_kv = nn.Linear(embed_dim, 2 * embed_dim)
        nn.init.trunc_normal_(self.part_queries, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        prefix = 1 + num_register_tokens
        rates = [drop_path_rate] * depth if drop_path_uniform else torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([ValidBlock(embed_dim, num_heads, rope=True, prefix=prefix, drop_path=r)
                                     for r in rates])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.detail_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.trunc_normal_(self.joint_pos, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)

    def get_config(self):
        return dict(self.model_config)

    def _prepare(self, clips, masks=None):
        valid = patch_validity(clips, self.patch_size[0])
        b, t, v = valid.shape
        detected = torch.isfinite(clips).all(1, keepdim=True) & (clips[:, 2:3] > 0)
        clean = torch.where(detected, clips, torch.zeros_like(clips))[..., 0]
        h = self.patch_embed(clean).reshape(b, t, v, self.embed_dim)
        if masks is not None:
            if masks.shape != (b, t * v):
                raise ValueError("artificial masks must use the original temporal-patch x joint grid")
            h = torch.where(masks.reshape(b, t, v, 1) & valid[..., None], self.mask_token.to(h.dtype), h)
        h = (h + self.joint_pos) * valid[..., None]
        parts = torch.stack([F.pad(h[:, :, a:z], (0, 0, 0, 21 - (z-a))) for a,z in PART_SLICES], 2)
        pv = torch.stack([F.pad(valid[:, :, a:z], (0, 21 - (z-a))) for a,z in PART_SLICES], 2)
        parts, pv_flat = parts.reshape(b*t*4, 21, self.embed_dim), pv.reshape(b*t*4, 21)
        for block in self.local_encoders:
            parts = block(parts, pv_flat)
        k, value = self.pool_kv(self.pool_norm(parts)).chunk(2, -1)
        safe = pv_flat.clone()
        safe[:, 0] |= ~pv_flat.any(1)
        queries = self.part_queries[None].expand(b*t, -1, -1, -1).reshape(b*t*4, -1, self.embed_dim)
        pooled = F.scaled_dot_product_attention(queries[:, None], k[:, None], value[:, None],
                                               attn_mask=safe[:, None, None]).squeeze(1)
        pooled = ((pooled + queries) * pv_flat.any(1)[:, None, None]).reshape(b, t, 4, -1, self.embed_dim)
        parts = parts.reshape(b, t, 4, 21, self.embed_dim)
        local = torch.cat([parts[:, :, i, :z-a] for i,(a,z) in enumerate(PART_SLICES)], 2)
        summaries = torch.cat([pooled[:, :, i, :k] for i,k in enumerate(self.part_tokens)], 2).flatten(1, 2)
        sv = torch.cat([pv[:, :, i].any(-1, keepdim=True).expand(-1, -1, k)
                        for i,k in enumerate(self.part_tokens)], 2).flatten(1)
        prefix = torch.cat((self.cls_token, self.register_tokens), 1).expand(b, -1, -1)
        seq = torch.cat((prefix, summaries), 1)
        seq_valid = torch.cat((torch.ones(b, prefix.shape[1], device=sv.device, dtype=torch.bool), sv), 1)
        return seq, seq_valid, local, valid

    def _features(self, seq, local, valid):
        b, t, v, d = local.shape
        seq = self.norm(seq)
        summaries = seq[:, 1 + self.num_register_tokens:].reshape(b, t, self.summaries_per_time, d)
        summary_mask = torch.cat([valid[:, :, a:z].any(-1, keepdim=True).expand(-1, -1, k)
                                  for (a,z),k in zip(PART_SLICES,self.part_tokens)], 2)
        summaries = summaries * summary_mask[..., None]
        contexts, pos = [], 0
        for (start, end), k in zip(PART_SLICES, self.part_tokens):
            contexts.append(summaries[:, :, pos:pos + k].mean(2, keepdim=True).expand(-1, -1, end - start, -1))
            pos += k
        detail = self.detail_norm(local + self.context_proj(torch.cat(contexts, 2))) * valid[..., None]
        # Equal weight per detected part; absent joints/parts never dilute the representation.
        means, usable = [], []
        for start, end in PART_SLICES:
            pv = valid[:, :, start:end]
            means.append(detail[:, :, start:end].sum((1, 2)) / pv.sum((1, 2)).clamp_min(1)[:, None])
            usable.append(pv.any(2).any(1))
        means, usable = torch.stack(means, 1), torch.stack(usable, 1)
        pooled = (means * usable[..., None]).sum(1) / usable.sum(1).clamp_min(1)[:, None]
        embedding = self.norm(seq[:, 0] + pooled)
        embedding = embedding * valid.any(2).any(1)[:, None]
        return dict(x_norm_clstoken=seq[:, 0], x_norm_regtokens=seq[:, 1:1 + self.num_register_tokens],
                    x_norm_patchtokens=detail.flatten(1, 2), patch_valid=valid.flatten(1),
                    x_norm_parttokens=summaries.flatten(1, 2), x_norm_embedding=embedding, x_prenorm=seq)

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            masks = masks if masks is not None else [None] * len(x)
            return [self.forward_features(a, m) for a, m in zip(x, masks)]
        seq, sv, local, valid = self._prepare(x, masks)
        all_valid = bool(sv.all())
        for block in self.blocks:
            seq = block(seq, sv, valid.shape[1], all_valid)
        out = self._features(seq, local, valid)
        out["masks"] = masks
        return out

    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        if not norm:
            raise ValueError("V2 intermediate joint features are normalized after context fusion")
        take = set(range(self.n_blocks - n, self.n_blocks)) if isinstance(n, int) else set(n)
        if not take or min(take) < 0 or max(take) >= self.n_blocks:
            raise ValueError("invalid intermediate layer selection")
        seq, sv, local, valid = self._prepare(x)
        outputs = []
        all_valid = bool(sv.all())
        for i, block in enumerate(self.blocks):
            seq = block(seq, sv, valid.shape[1], all_valid)
            if i in take:
                f = self._features(seq, local, valid)
                patches = f["x_norm_patchtokens"]
                if reshape:
                    patches = patches.reshape(x.shape[0], valid.shape[1], 69, self.embed_dim).permute(0, 3, 1, 2)
                outputs.append((patches, f["x_norm_clstoken"]) if return_class_token else patches)
        return tuple(outputs)

    def save_pretrained(self, save_directory, config=None):
        from pathlib import Path
        import json
        from safetensors.torch import save_file
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        if config and config != self.get_config():
            raise ValueError("export config must describe this exact V2 encoder")
        (root / "config.json").write_text(json.dumps(self.get_config(), indent=2) + "\n")
        save_file({k: v.contiguous() for k, v in self.state_dict().items()}, str(root / "model.safetensors"))

    def forward(self, x, masks=None, is_training=False):
        features = self.forward_features(x, masks)
        return features if is_training else features["x_norm_embedding"]
