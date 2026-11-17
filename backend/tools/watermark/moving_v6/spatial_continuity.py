"""Mask-bounded moving-watermark repair using independent spatial evidence.

The caller supplies a color image before spatial synthesis. ``write_mask`` is
the only writable region; ``evidence_mask`` can additionally exclude temporal
estimates or uncertain observations without authorizing any extra writes.
"""
from functools import lru_cache
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .spatial_fallback import spatial_fallback as legacy_fallback
from .exemplar_texture import synthesize


@lru_cache(maxsize=1)
def _core():
    # The preserved R16 modules use absolute imports. Load them lazily so the
    # process can prepare optional native libraries before their first import.
    directory = str(Path(__file__).resolve().parents[1] / "fixed_r16" / "v6_core")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from spatial_fallback import spatial_fallback
    from edge_continuity import edge_continuity
    from repair_evidence import merge_generated
    from affine_postprocess import repair as affine_repair
    from tensor_harmonic import tensor_field
    return spatial_fallback, edge_continuity, merge_generated, affine_repair, tensor_field


def clean_evidence_mask(patch, mask):
    """Reject narrow fringe outliers in any color channel or polarity.

    An adjacent sample is excluded only when it lies outside the color envelope
    of independently observed outer samples. Continued bright, dark and colored
    edges with outer support remain available. This mask never permits writes.
    """
    unknown = np.asarray(mask, bool)
    if not unknown.any():
        return unknown.copy(), 0
    distance = cv2.distanceTransform(unknown.astype(np.uint8), cv2.DIST_L2, 5)
    radius = int(np.clip(np.rint(np.percentile(distance[unknown], 85) * .15), 1, 2))
    kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
    guard = cv2.dilate(unknown.astype(np.uint8), kernel) > 0
    # Look beyond short antialiased glyph endpoints before calling a sample an
    # isolated fringe. The exclusion itself remains only one or two pixels wide.
    span = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (radius + 6) + 1,) * 2)
    observed = ~guard
    upper = cv2.dilate(np.where(observed[..., None], patch, 0).astype(np.uint8), span)
    lower = cv2.erode(np.where(observed[..., None], patch, 255).astype(np.uint8), span)
    support = cv2.dilate(observed.astype(np.uint8), span) > 0
    foreground_upper = cv2.dilate(np.where(unknown[..., None], patch, 0).astype(np.uint8), span)
    foreground_lower = cv2.erode(np.where(unknown[..., None], patch, 255).astype(np.uint8), span)
    colors = patch.astype(np.int16)
    # A bright target must not discard an adjacent dark text endpoint, nor a
    # dark target a bright endpoint. Require the same local color direction.
    above = (colors > upper.astype(np.int16) + 6) & (colors <= foreground_upper.astype(np.int16))
    below = (colors < lower.astype(np.int16) - 6) & (colors >= foreground_lower.astype(np.int16))
    outlier = np.any(above | below, axis=2)
    return unknown | (guard & support & outlier), radius


def _unsupported_components(image, exclusion, write, flat, tensor_field):
    """Find holes without enough evidence for planar or directional repair."""
    if not np.any(write & (flat == 0)):
        return np.zeros(write.shape, bool), 0
    _, confidence, _, _ = tensor_field(image, exclusion, write)
    count, labels = cv2.connectedComponents(write.astype(np.uint8), 8)
    remaining = write & (flat == 0)
    sizes = np.bincount(labels[remaining], minlength=count)
    supported = np.bincount(labels[remaining & (confidence >= .25)], minlength=count)
    # q combines independent observation coverage, local flatness and squared
    # directional coherence. At full coverage, textured structure needs at
    # least 0.5 coherence to justify a directional replacement. Require a
    # majority of the hole to meet that support: a few strong boundary edges
    # must not lift an average over a mostly unsupported textured interior.
    rejected_components = (sizes > 0) & (supported * 2 <= sizes)
    rejected_components[0] = False
    return remaining & rejected_components[labels], int(rejected_components.sum())


def repair(patch, write_mask, *, evidence_mask=None, edge_enabled=True):
    """Return ``(image, diagnostics)`` while preserving every non-write pixel.

    All fits receive the same input observations and exclusion mask. Generated
    planes, edge proposals and solver output are never fed back as independent
    observations. Native preparation belongs to the worker startup, not here.
    """
    started = time.perf_counter()
    image = np.asarray(patch)
    write = np.asarray(write_mask, bool)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a uint8 HWC color image")
    if write.ndim != 2 or write.shape != image.shape[:2]:
        raise ValueError("Expected an HW write mask matching the image")
    evidence = write.copy() if evidence_mask is None else np.asarray(evidence_mask, bool)
    if evidence.shape != write.shape:
        raise ValueError("Expected an evidence mask matching the write mask")
    evidence = evidence | write
    statistics = dict(write_pixels=int(write.sum()), changed_pixels=0, flat_pixels=0,
                      edge_pixels=0, seam_pixels=0, texture_pixels=0, evidence_guard_radius=0,
                      unsupported_pixels=0, unsupported_components=0,
                      guard_unsupported_components=0,
                      excluded_reference_pixels=int((evidence & ~write).sum()),
                      background_continuity={}, edge_continuity={}, affine_postprocess={})
    if not write.any():
        statistics["seconds"] = time.perf_counter() - started
        return image.copy(), statistics
    spatial_fallback, edge_continuity, merge_generated, affine_repair, tensor_field = _core()
    exclusion, radius = clean_evidence_mask(image, evidence)
    guard_unsupported_count = 0
    if np.any(exclusion & ~evidence):
        unsupported_guard, guard_unsupported_count = _unsupported_components(
            image, exclusion, evidence, np.zeros(write.shape, np.uint8), tensor_field)
        if unsupported_guard.any():
            kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
            protected = cv2.dilate(unsupported_guard.astype(np.uint8), kernel) > 0
            # In unresolved text/texture, a same-color endpoint is ambiguous.
            # Keep that observation rather than declaring it watermark fringe.
            exclusion = evidence | (exclusion & ~protected)
    generated, flat, background_stats = spatial_fallback(
        image, exclusion, write_mask=write, diagnostics=True)
    unsupported, unsupported_count = _unsupported_components(image, exclusion, write, flat, tensor_field)
    if unsupported.any():
        conservative, _ = legacy_fallback(image, exclusion)
        generated[unsupported] = conservative[unsupported]
    # A boundary tensor can be coherent while a large hidden interior contains
    # a different edge or texture. Geometry and held-out visible plane checks
    # decide whether exemplars are applicable, independent of that tensor vote.
    textured, selected, texture_stats = synthesize(image, write & (flat == 0), exclusion)
    generated[selected] = textured[selected]
    statistics['texture_pixels'] = int(selected.sum())
    statistics['texture_synthesis'] = texture_stats
    edge_stats = {}
    accepted = np.zeros(write.shape, bool)
    seam_pixels = 0
    if edge_enabled and np.any(write & (flat == 0)):
        corrected, edge_mask, edge_stats = edge_continuity(image, exclusion, generated)
        accepted = edge_mask & (flat == 0) & write
        generated, seam_pixels = merge_generated(
            generated, corrected, accepted, write, evidence_mask=exclusion)
    generated, affine_stats = affine_repair(image, exclusion, write, generated)
    result = image.copy()
    result[write] = generated[write]
    statistics.update(changed_pixels=int(np.any(result != image, axis=2)[write].sum()),
                      flat_pixels=int((flat.astype(bool) & write).sum()),
                      edge_pixels=int(accepted.sum()), seam_pixels=seam_pixels,
                      unsupported_pixels=int(unsupported.sum()),
                      unsupported_components=unsupported_count,
                      guard_unsupported_components=guard_unsupported_count,
                      evidence_guard_radius=radius,
                      excluded_reference_pixels=int((exclusion & ~write).sum()),
                      background_continuity=background_stats, edge_continuity=edge_stats,
                      affine_postprocess=affine_stats, seconds=time.perf_counter() - started)
    return result, statistics
