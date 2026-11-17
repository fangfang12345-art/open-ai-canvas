"""Calibrate a stationary light overlay inside a caller-selected search region.

This identifies persistent, bright neutral components, not logo text. The caller
must select the intended region. A persistent bright scene feature can satisfy
the same statistics, so every generated mask includes a review image.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import json
import subprocess
from pathlib import Path

from media import executable

import cv2
import numpy as np


def probe(source):
    data = json.loads(subprocess.check_output([
        executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,duration',
        '-of', 'json', str(source),
    ]))['streams'][0]
    a, b = map(float, data['avg_frame_rate'].split('/'))
    return int(data['width']), int(data['height']), a/b, float(data['duration'])


def sample_patches(source, seconds, roi, sample_rate=3.):
    x0, y0, x1, y1 = roi
    width, height = x1-x0, y1-y0
    command = [
        executable('ffmpeg'), '-nostdin', '-v', 'error', '-threads', '2',
        '-i', str(source), '-t', str(seconds), '-an', '-vf',
        f'crop={width}:{height}:{x0}:{y0},fps={sample_rate}',
        '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-',
    ]
    raw = subprocess.check_output(command)
    frame_bytes = width*height*3
    if len(raw) % frame_bytes or len(raw)//frame_bytes < 3:
        raise ValueError('Insufficient or malformed calibration frames')
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3).copy()


def complete_light_fringe(samples, support, core):
    """Add only persistent white-mixture evidence in a scale-bounded fringe.

    An antialiased white overlay inherits background chroma, so the core's
    neutrality test is inappropriate here. Estimate a background from outside
    the entire candidate band, then require positive, temporally consistent
    movement toward white. This is mask evidence, not background restoration.
    """
    support = np.asarray(support, bool)
    core = np.asarray(core, bool)
    if samples.shape[1:3] != support.shape or support.shape != core.shape:
        raise ValueError('Samples, support, and core dimensions must agree')
    if not core.any():
        return support.copy(), dict(added_pixels=0, candidate_pixels=0)
    thickness = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    stroke_radius = float(np.percentile(thickness[core], 75))
    fringe_radius = int(np.clip(np.ceil(stroke_radius*.25), 1, 6))
    background_radius = max(5, int(np.ceil(stroke_radius*1.25)))
    kernel_size = 2*background_radius+1
    yy, xx = np.nonzero(support)
    margin = background_radius+fringe_radius+1
    left, top = max(0, xx.min()-margin), max(0, yy.min()-margin)
    right = min(support.shape[1], xx.max()+margin+1)
    bottom = min(support.shape[0], yy.max()+margin+1)
    local = np.s_[top:bottom, left:right]
    local_support = support[local]
    excluded = cv2.dilate(local_support.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (2*fringe_radius+1,)*2)) > 0
    cy, cx = np.nonzero(excluded & ~local_support)
    stats = dict(added_pixels=0, evidence_pixels=0, disconnected_pixels=0,
                 candidate_pixels=len(cy),
                 stroke_radius=stroke_radius, fringe_radius=fringe_radius,
                 background_radius=background_radius)
    if not len(cy):
        return support.copy(), stats
    valid = (~excluded).astype(np.float32)
    weight = cv2.boxFilter(valid, -1, (kernel_size,)*2, normalize=False,
                          borderType=cv2.BORDER_CONSTANT)
    enough_background = weight[cy, cx] >= max(8, kernel_size*kernel_size*.12)
    votes, alphas = [], []
    for frame in samples:
        observed_patch = frame[local].astype(np.float32)
        sums = cv2.boxFilter(observed_patch*valid[:, :, None], -1,
                            (kernel_size,)*2, normalize=False,
                            borderType=cv2.BORDER_CONSTANT)
        background = sums[cy, cx]/np.maximum(weight[cy, cx, None], 1.)
        observed = observed_patch[cy, cx]
        toward_white = 255.-background
        difference = observed-background
        alpha = np.sum(difference*toward_white, axis=1)/np.maximum(
            np.sum(toward_white*toward_white, axis=1), 1.)
        lift = np.mean(difference, axis=1)
        residual = np.max(abs(difference-alpha[:, None]*toward_white), axis=1)
        # A small absolute tolerance covers quantization/chroma filtering;
        # the relative term covers imperfect local background approximation.
        votes.append((alpha >= .12) & (alpha <= 1.1) & (lift >= 12.)
                     & (residual <= 8.+.3*lift))
        alphas.append(alpha)
    alphas = np.asarray(alphas)
    spread = np.quantile(alphas, .9, axis=0)-np.quantile(alphas, .1, axis=0)
    accepted = enough_background & (np.mean(votes, axis=0) >= .8) & (spread <= .45)
    local_result = local_support.copy()
    local_result[cy[accepted], cx[accepted]] = True
    _, labels = cv2.connectedComponents(local_result.astype(np.uint8), 8)
    attached = np.isin(labels, np.unique(labels[local_support])) & (labels != 0)
    # Presence is verified per core-bearing component. Do not report isolated
    # fringe islands that cannot inherit verification, or invent bridging pixels.
    result = support.copy()
    result[local] = attached
    stats['evidence_pixels'] = int(accepted.sum())
    stats['added_pixels'] = int(np.count_nonzero(attached & ~local_support))
    stats['disconnected_pixels'] = stats['evidence_pixels']-stats['added_pixels']
    return result, stats


def detect_persistent_light(samples, *, brightness=170., edge_brightness=12.,
                            low_quantile=.1, max_chroma=22., min_area=12,
                            edge_radius=5, halo=2):
    if samples.ndim != 4 or samples.shape[-1] != 3:
        raise ValueError('Expected N x H x W x 3 BGR frames')
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in samples])
    chroma = samples.max(axis=3).astype(np.int16)-samples.min(axis=3).astype(np.int16)
    low = np.quantile(gray, low_quantile, axis=0).astype(np.float32)
    neutral = np.quantile(chroma, 1-low_quantile, axis=0) <= max_chroma
    core = (low >= brightness) & neutral
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    kept = np.zeros_like(core)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            kept |= labels == index
    if not kept.any():
        raise ValueError('No persistent bright neutral watermark candidates')
    near = cv2.dilate(kept.astype(np.uint8), np.ones((2*edge_radius+1,)*2, np.uint8)) > 0
    edges = near & (low >= edge_brightness) & neutral
    count, labels = cv2.connectedComponents(edges.astype(np.uint8), 8)
    connected = np.isin(labels, np.unique(labels[kept])) & (labels != 0)
    support = cv2.dilate(connected.astype(np.uint8), np.ones((2*halo+1,)*2, np.uint8)) > 0
    support, fringe = complete_light_fringe(samples, support, kept)
    return support, kept, low, dict(
        brightness=brightness, edge_brightness=edge_brightness,
        low_quantile=low_quantile, max_chroma=max_chroma,
        min_component_area=min_area, edge_radius=edge_radius, halo=halo,
        raw_components=count-1, core_pixels=int(kept.sum()),
        mask_pixels=int(support.sum()),
        fringe_completion=fringe,
    )


def present_mask(patch, support, core, *, brightness=150., max_chroma=35.,
                 min_fraction=.85, min_contrast=20., ring_radius=8):
    """Reject missing components without disabling unrelated visible components.

    This is a conservative presence gate for the calibrated light overlay. It
    does not track an animated entrance, and cannot disambiguate identical scene
    shapes. Call it on every target and every donor before exposing their masks.
    """
    support, core = np.asarray(support, bool), np.asarray(core, bool)
    if patch.shape[:2] != support.shape or support.shape != core.shape:
        raise ValueError('Patch, support, and core dimensions must agree')
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    chroma = patch.max(axis=2).astype(np.int16)-patch.min(axis=2).astype(np.int16)
    bright = (gray >= brightness) & (chroma <= max_chroma)
    count, labels = cv2.connectedComponents(support.astype(np.uint8), 8)
    accepted = np.zeros_like(support)
    reports = []
    for index in range(1, count):
        component = labels == index
        seed = component & core
        if not seed.any():
            continue
        ring = (cv2.dilate(component.astype(np.uint8), np.ones((2*ring_radius+1,)*2, np.uint8)) > 0) & ~support
        fraction = float(bright[seed].mean())
        contrast = float(np.median(gray[seed])-np.median(gray[ring])) if ring.any() else 0.
        keep = fraction >= min_fraction and contrast >= min_contrast
        if keep:
            accepted |= component
        reports.append(dict(component=index, pixels=int(component.sum()), core_pixels=int(seed.sum()),
                            bright_fraction=fraction, local_contrast=contrast, accepted=keep))
    return accepted, dict(components=reports,
                          unverified_components=[x['component'] for x in reports if not x['accepted']],
                          donor_safe=bool(reports) and all(x['accepted'] for x in reports),
                          verified_pixels=int(accepted.sum()),
                          reason='all_calibrated_components_present' if reports and all(x['accepted'] for x in reports)
                          else 'uncertain_components_must_not_be_used_as_clean_donor_pixels')


_PresenceGeometry = namedtuple('_PresenceGeometry', 'shape ring_radius support core components')
_ComponentGeometry = namedtuple('_ComponentGeometry', 'index component_indices seed_indices ring_indices')


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def compile_presence_geometry(support, core, *, ring_radius=8):
    """Own fixed geometry; rebuild when a mask, core, or ring radius changes."""
    support = _readonly(np.asarray(support, bool))
    core = _readonly(np.asarray(core, bool))
    if support.ndim != 2 or support.shape != core.shape:
        raise ValueError('Support and core must be equally shaped 2-D masks')
    if ring_radius < 0 or int(ring_radius) != ring_radius:
        raise ValueError('Ring radius must be a nonnegative integer')
    count, labels = cv2.connectedComponents(support.astype(np.uint8), 8)
    kernel = np.ones((2*int(ring_radius)+1,)*2, np.uint8)
    components = []
    for index in range(1, count):
        component = labels == index
        seed = component & core
        if not seed.any():
            continue
        ring = (cv2.dilate(component.astype(np.uint8), kernel) > 0) & ~support
        components.append(_ComponentGeometry(
            index, _readonly(np.flatnonzero(component)),
            _readonly(np.flatnonzero(seed)), _readonly(np.flatnonzero(ring)),
        ))
    return _PresenceGeometry(support.shape, int(ring_radius), support, core, tuple(components))


def present_mask_prepared(patch, geometry, *, brightness=150., max_chroma=35.,
                          min_fraction=.85, min_contrast=20., ring_radius=8):
    if patch.shape[:2] != geometry.shape:
        raise ValueError('Patch and prepared mask dimensions must agree')
    if ring_radius != geometry.ring_radius:
        raise ValueError('Ring radius changed: reload calibration or rebuild presence geometry')
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    chroma = patch.max(axis=2).astype(np.int16)-patch.min(axis=2).astype(np.int16)
    bright = (gray >= brightness) & (chroma <= max_chroma)
    gray_flat, bright_flat = gray.ravel(), bright.ravel()
    accepted = np.zeros(geometry.shape, dtype=bool)
    accepted_flat = accepted.ravel()
    reports = []
    for component in geometry.components:
        fraction = float(bright_flat[component.seed_indices].mean())
        contrast = float(np.median(gray_flat[component.seed_indices])
                         - np.median(gray_flat[component.ring_indices])) if len(component.ring_indices) else 0.
        keep = fraction >= min_fraction and contrast >= min_contrast
        if keep:
            accepted_flat[component.component_indices] = True
        reports.append(dict(component=component.index,
                            pixels=len(component.component_indices),
                            core_pixels=len(component.seed_indices),
                            bright_fraction=fraction, local_contrast=contrast,
                            accepted=keep))
    return accepted, dict(components=reports,
                          unverified_components=[x['component'] for x in reports if not x['accepted']],
                          donor_safe=bool(reports) and all(x['accepted'] for x in reports),
                          verified_pixels=int(accepted.sum()),
                          reason='all_calibrated_components_present' if reports and all(x['accepted'] for x in reports)
                          else 'uncertain_components_must_not_be_used_as_clean_donor_pixels')


def load_calibration(calibration_dir):
    directory = Path(calibration_dir)
    model = json.loads((directory/'config.json').read_text())
    model['mask'] = np.load(directory/'mask.npy', allow_pickle=False).astype(bool)
    model['core'] = np.load(directory/'core.npy', allow_pickle=False).astype(bool)
    x0,y0,x1,y1 = model['roi']
    if model['mask'].shape != (y1-y0,x1-x0) or model['core'].shape != model['mask'].shape:
        raise ValueError('Calibration arrays disagree with ROI')
    if np.any(model['core'] & ~model['mask']):
        raise ValueError('Core must be contained by watermark support')
    geometry = compile_presence_geometry(model['mask'], model['core'],
                                         ring_radius=model['presence_gate'].get('ring_radius', 8))
    model['mask'], model['core'] = geometry.support, geometry.core
    model['_presence_geometry'] = geometry
    return model


def frame_mask(frame_bgr, model):
    if frame_bgr.shape[:2] != (model['source_height'],model['source_width']):
        raise ValueError('Frame dimensions differ from calibration source')
    x0,y0,x1,y1 = model['roi']
    geometry = model.get('_presence_geometry')
    if geometry is None:
        mask, info = present_mask(frame_bgr[y0:y1,x0:x1], model['mask'], model['core'],
                                  **model['presence_gate'])
    else:
        if (model['mask'] is not geometry.support or model['core'] is not geometry.core
                or model['mask'].flags.writeable or model['core'].flags.writeable):
            raise ValueError('Calibration mask or core changed: reload calibration')
        mask, info = present_mask_prepared(frame_bgr[y0:y1,x0:x1], geometry,
                                           **model['presence_gate'])
    return list(model['roi']), mask, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--normalized-roi', default='0,0,0.5,0.25')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--sample-rate', type=float, default=3.)
    parser.add_argument('--halo', type=int, default=2)
    args = parser.parse_args()
    width, height, fps, duration = probe(args.source)
    box = list(map(float, args.normalized_roi.split(',')))
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        parser.error('normalized-roi must be x0,y0,x1,y1 inside [0,1]')
    if args.seconds <= 0 or args.seconds > duration or args.sample_rate <= 0 or args.halo < 0:
        parser.error('Invalid duration, sampling rate, or halo')
    # Chroma-subsampled decoding requires even crop origin and extent.
    roi = [int(box[0]*width)//2*2, int(box[1]*height)//2*2,
           int(np.ceil(box[2]*width/2))*2, int(np.ceil(box[3]*height/2))*2]
    samples = sample_patches(args.source, args.seconds, roi, args.sample_rate)
    support, core, template, statistics = detect_persistent_light(samples, halo=args.halo)
    yy, xx = np.nonzero(support)
    margin = 10
    left, top = max(0, xx.min()-margin), max(0, yy.min()-margin)
    right, bottom = min(support.shape[1], xx.max()+margin+1), min(support.shape[0], yy.max()+margin+1)
    local = np.s_[top:bottom, left:right]
    final_roi = [roi[0]+int(left), roi[1]+int(top), roi[0]+int(right), roi[1]+int(bottom)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir/'mask.npy', support[local])
    np.save(args.out_dir/'core.npy', core[local])
    for name, data in [('mask.png', support[local].astype(np.uint8)*255),
                       ('core.png', core[local].astype(np.uint8)*255),
                       ('template.png', np.rint(template[local]).astype(np.uint8))]:
        if not cv2.imwrite(str(args.out_dir/name), data):
            raise IOError(name)
    preview = samples[len(samples)//2].copy()
    preview[support] = (preview[support].astype(np.float32)*.35+np.float32([20,30,255])*.65).astype(np.uint8)
    panels = np.concatenate([samples[0], samples[len(samples)//2], preview,
                             cv2.cvtColor(support.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)],axis=0)
    cv2.imwrite(str(args.out_dir/'mask-review.png'), panels)
    config = dict(
        source=str(args.source.resolve()), source_domain='original_raw_video',
        source_width=width, source_height=height, fps=fps,
        seconds=args.seconds, calibration_interval=[0,args.seconds],
        sample_rate=args.sample_rate, sample_count=len(samples),
        normalized_search_roi=box, search_roi=roi, roi=final_roi,
        mask=str((args.out_dir/'mask.npy').resolve()),
        core=str((args.out_dir/'core.npy').resolve()),
        mask_shape=list(support[local].shape), stationary=True,
        strategy='persistent_light_core_with_evidence_bounded_colored_fringe',
        statistics=statistics,
        limitations=['Requires a caller-selected watermark search region.',
                     'Persistent bright background features may be indistinguishable.',
                     'A mask does not recover background hidden by opaque pixels.'],
    )
    presence = []
    for index, sample in enumerate(samples):
        accepted, presence_info = present_mask(sample[local], support[local], core[local])
        presence.append(dict(sample_index=index,
                             approximate_time=(index+.5)/args.sample_rate,
                             accepted_pixels=int(accepted.sum()),
                             calibrated_pixels=int(support[local].sum()),
                             **presence_info))
    config['presence_gate'] = dict(brightness=150., max_chroma=35., min_fraction=.85,
                                   min_contrast=20., ring_radius=8)
    config['calibration_presence_min_pixels'] = min(x['accepted_pixels'] for x in presence)
    config['calibration_presence_max_pixels'] = max(x['accepted_pixels'] for x in presence)
    config['presence_report'] = str((args.out_dir/'presence-samples.json').resolve())
    (args.out_dir/'presence-samples.json').write_text(json.dumps(presence,ensure_ascii=False,indent=2)+'\n')
    (args.out_dir/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(config,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
