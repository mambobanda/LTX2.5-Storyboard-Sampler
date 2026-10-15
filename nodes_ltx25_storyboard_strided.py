"""LTX2.5 Storyboard Sampler (strided timeline, single chunk).

Generates N sparse panels in one pass by conditioning at frame_rate = fps / stride,
re-anchors the reference image with guide keyframes so it is not ignored at large
strides, and decodes panels directly to IMAGE.

v2 changes:
- separate latent strength vs attention strength for guides (attention_scale)
- re-anchor schedules (constant / linear / cosine decay) + min_anchor_gap + anchor_window
- sharpest-frame selection inside each latent's pixel window (fixes blurry panels)
- optional unsharp mask and crisp-prompt suffix
"""

import math
import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import io

try:
    from comfy_extras.nodes_lt import LTXVAddGuide, _append_guide_attention_entry
    HAS_LTX_HELPERS = True
except Exception:
    LTXVAddGuide = _append_guide_attention_entry = None
    HAS_LTX_HELPERS = False

MAX_STRIDE = 24
CRISP_SUFFIX = " sharp focus, crisp detail, static camera, no motion blur, high quality."


def _log(on, msg):
    if on:
        print(f"[LTX25StoryboardStrided] {msg}")


def _encode_ref(vae, image, w, h, device, dtype):
    img = image[:1]
    if img.shape[1] != h or img.shape[2] != w:
        img = comfy.utils.common_upscale(img.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)
    t = vae.encode(img[:, :, :, :3])
    if t.ndim == 4:
        t = t.unsqueeze(2)
    return t.to(device=device, dtype=dtype)


def _add_guide(positive, negative, samples, mask, t, pixel_index, strength, attn_scale, scale_factors, dbg):
    frame_idx, lat_idx = LTXVAddGuide.get_latent_index(
        positive, samples.shape[2], t.shape[0], pixel_index, scale_factors, latent_shape=samples.shape
    )
    positive, negative, samples, mask = LTXVAddGuide.append_keyframe(
        positive, negative, frame_idx, samples, mask, t, strength, scale_factors
    )
    positive, negative = _append_guide_attention_entry(
        positive, negative, t.shape[2] * t.shape[3] * t.shape[4], list(t.shape[2:]),
        strength=float(strength) * float(attn_scale),
    )
    _log(dbg, f"guide pixel_idx={pixel_index} frame_idx={frame_idx} latent_idx={lat_idx} "
              f"strength={strength:.3f} attn={strength * attn_scale:.3f}")
    return positive, negative, samples, mask


def _schedule(kind, k, n):
    x = k / max(1, n - 1)
    if kind == "linear_decay":
        return 1.0 - x
    if kind == "cosine_decay":
        return 0.5 * (1.0 + math.cos(math.pi * x))
    return 1.0


def _sharpness(frames):
    g = frames[..., :3].float().mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=g.device).view(1, 1, 3, 3)
    lap = F.conv2d(g, k, padding=1)
    return lap.flatten(1).var(dim=1)


def _unsharp(img, amount):
    if amount <= 0.0:
        return img
    x = img.permute(0, 3, 1, 2)
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    out = x + amount * (x - blur)
    return out.clamp(0.0, 1.0).permute(0, 2, 3, 1)


def _pick_frames(frames, num_panels, ts, mode):
    total = frames.shape[0]
    picks = []
    for k in range(num_panels):
        if k == 0:
            picks.append(0)
            continue
        lo = min(ts * (k - 1) + 1, total - 1)
        hi = min(ts * k, total - 1)
        if mode == "last":
            picks.append(hi)
        elif mode == "center":
            picks.append((lo + hi) // 2)
        else:
            win = frames[lo:hi + 1]
            picks.append(lo + int(_sharpness(win).argmax().item()))
    return picks


class LTX25StoryboardStrided(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25StoryboardStrided",
            display_name="LTX2.5 Storyboard Sampler (strided, single chunk)",
            category="model/sampling/ltx25",
            description="Sparse storyboard panels across the timeline with reference re-anchoring via guide keyframes. Experimental.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Clip.Input("clip"),
                io.Image.Input("reference_image", optional=True),
                io.String.Input("global_prompt", multiline=True, default=""),
                io.String.Input("panel_prompts", multiline=True, default="",
                                tooltip="One line per panel, joined into a shot sequence."),
                io.String.Input("negative_prompt", multiline=True,
                                default="worst quality, blurry, motion blur, out of focus, distorted"),
                io.Boolean.Input("crisp_prompt_suffix", default=True,
                                 tooltip="Append sharp-focus / no-motion-blur wording to the prompt."),
                io.Int.Input("num_panels", default=9, min=2, max=64),
                io.Int.Input("stride", default=8, min=1, max=MAX_STRIDE,
                             tooltip="Timeline spacing between panels in latent frames (max 24)."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0),
                io.Int.Input("width", default=768, min=64, max=4096, step=32),
                io.Int.Input("height", default=512, min=64, max=4096, step=32),
                io.Int.Input("latent_channels", default=128, min=4, max=256),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=8, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Input("ref_strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Strength of the frame-0 reference (pin + guide)."),
                io.Float.Input("attention_scale", default=0.6, min=0.0, max=1.0, step=0.05,
                               tooltip="Multiplier on guide attention strength (applies to frame-0 guide and re-anchors)."),
                io.Int.Input("reinject_every", default=0, min=0, max=64,
                             tooltip="Append the reference as an extra guide every N panels. 0 = off."),
                io.Float.Input("reinject_strength", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("reinject_schedule", options=["cosine_decay", "linear_decay", "constant"],
                               default="cosine_decay"),
                io.Int.Input("anchor_window", default=0, min=0, max=64,
                             tooltip="Only re-anchor within the first N panels (0 = whole timeline)."),
                io.Int.Input("min_anchor_gap", default=2, min=1, max=64,
                             tooltip="Minimum panels between re-anchors."),
                io.Combo.Input("frame_pick", options=["sharpest", "center", "last"], default="sharpest",
                               tooltip="Which pixel frame of each latent's window becomes the panel."),
                io.Float.Input("sharpen", default=0.0, min=0.0, max=2.0, step=0.05,
                               tooltip="Unsharp mask amount applied to output panels."),
                io.Int.Input("debug_verbose", default=1, min=0, max=1),
            ],
            outputs=[io.Image.Output(display_name="panels")],
        )

    @classmethod
    def execute(cls, model, vae, clip, global_prompt, panel_prompts, negative_prompt, crisp_prompt_suffix,
                num_panels, stride, fps, width, height, latent_channels, seed, steps, cfg,
                sampler_name, scheduler, ref_strength, attention_scale, reinject_every, reinject_strength,
                reinject_schedule, anchor_window, min_anchor_gap, frame_pick, sharpen, debug_verbose,
                reference_image=None):
        dbg = bool(debug_verbose)
        stride = min(int(stride), MAX_STRIDE)

        try:
            scale_factors = vae.downscale_index_formula
        except Exception:
            scale_factors = (8, 32, 32)
        ts = int(scale_factors[0])

        lines = [l.strip() for l in panel_prompts.splitlines() if l.strip()]
        text = global_prompt
        if lines:
            text += " Sequence of shots: " + " Then: ".join(lines)
        if crisp_prompt_suffix:
            text += CRISP_SUFFIX

        cond_fps = float(fps) / float(stride)

        def enc(s):
            tokens = clip.tokenize(s)
            c = clip.encode_from_tokens_scheduled(tokens)
            return node_helpers.conditioning_set_values(c, {"frame_rate": cond_fps})

        positive, negative = enc(text), enc(negative_prompt)

        samples = torch.zeros([1, latent_channels, num_panels, height // 32, width // 32])
        mask = torch.ones((1, 1, num_panels, 1, 1), dtype=samples.dtype)

        if reference_image is not None:
            if not HAS_LTX_HELPERS:
                raise RuntimeError("comfy_extras.nodes_lt helpers not found; cannot add guides.")
            t = _encode_ref(vae, reference_image, width, height, samples.device, samples.dtype)

            n0 = min(t.shape[2], num_panels)
            samples[:, :, :n0] = t[:, :, :n0]
            mask[:, :, :n0] = 1.0 - ref_strength

            positive, negative, samples, mask = _add_guide(
                positive, negative, samples, mask, t, 0, ref_strength, attention_scale, scale_factors, dbg)

            if reinject_every > 0:
                last_k = 0
                limit = num_panels if anchor_window <= 0 else min(anchor_window, num_panels)
                for k in range(reinject_every, limit, reinject_every):
                    if k - last_k < min_anchor_gap:
                        continue
                    s = float(reinject_strength) * _schedule(reinject_schedule, k, limit)
                    if s <= 0.01:
                        continue
                    positive, negative, samples, mask = _add_guide(
                        positive, negative, samples, mask, t, k * ts, s, attention_scale, scale_factors, dbg)
                    last_k = k

        _log(dbg, f"latent={tuple(samples.shape)} cond_fps={cond_fps:.4f} stride={stride}")

        lat = {"samples": samples, "noise_mask": mask}

        out = nodes.common_ksampler(model, int(seed), steps, cfg, sampler_name, scheduler,
                                    positive, negative, lat, denoise=1.0)[0]["samples"]

        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"Expected tensor latent from sampler, got {type(out)}")
        if out.ndim != 5:
            raise RuntimeError(f"Expected 5D latent [B,C,T,H,W], got shape {tuple(out.shape)}")

        out = out[:, :, :num_panels]
        frames = vae.decode(out)
        if frames.dim() == 5:
            frames = frames.reshape(-1, *frames.shape[-3:])

        idx = _pick_frames(frames, num_panels, ts, frame_pick)
        _log(dbg, f"picked pixel frames: {idx}")
        panels = _unsharp(frames[idx], float(sharpen))
        return io.NodeOutput(panels)
