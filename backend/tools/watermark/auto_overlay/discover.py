"""Discover stationary repeated contour proposals without a supplied ROI.

Short and long temporal windows preserve intermittent overlays. Repeated scene
objects can pass these proposal tests, so every output remains unverified until
independent identity/activity and background-motion evidence are available.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media import Reader, probe


def source_signature(source, info):
    path = Path(source).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtimeNs=stat.st_mtime_ns,
                width=info['width'], height=info['height'], frames=info['frames'])


def sample_source(source, info, output, budget=96, max_side=1280):
    signature = source_signature(source, info)
    scale = min(1., max_side / max(info['width'], info['height']))
    width, height = round(info['width'] * scale), round(info['height'] * scale)
    indices = np.unique(np.linspace(0, info['frames']-1, min(budget, info['frames'])).round().astype(int))
    cache = output / 'samples.npz'
    key = dict(source=signature, indices=indices.tolist(), size=[width, height])
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved['key'])) == key:
                return saved['gray'].copy(), indices, scale
    frames = []
    reader = Reader(source)
    try:
        for number, index in enumerate(indices):
            frame = reader.frame(int(index))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA))
            if number % 12 == 0:
                print(json.dumps(dict(stage='sampling', completed=number+1, total=len(indices))), flush=True)
    finally:
        reader.close()
    gray = np.stack(frames)
    np.savez_compressed(cache, gray=gray, frameIndices=indices, scale=scale,
                        size=np.array([info['width'], info['height']]), key=json.dumps(key))
    return gray, indices, scale


def contrast(frame):
    # Both polarities are retained. This measures local contrast, not whiteness,
    # and therefore does not assume neutral/white foreground pixels.
    frame = cv2.GaussianBlur(frame, (3, 3), .6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel)
    return np.maximum(bright, dark)


def temporal_windows(length):
    windows = set()
    size = 3
    while size < length:
        step = max(1, size // 2)
        windows.update((start, min(length, start+size)) for start in range(0, length-size+1, step))
        size *= 2
    windows.add((0, length))
    return sorted(windows)


def boxes_from_mask(mask):
    joined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    components = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area < 16 or min(width, height) < 3:
            continue
        # Long thin scene rules are not independently identifiable glyphs.
        if max(width/height, height/width) > 18:
            continue
        if width * height > mask.size * .12:
            continue
        components.append(dict(box=[x,y,x+width,y+height], label=label))
    # Group neighboring strokes into objects/lines, keeping independent groups.
    groups = []
    for component in sorted(components, key=lambda c: c['box'][0]):
        x0,y0,x1,y1 = component['box']
        merged = []
        for i, group in enumerate(groups):
            a,b,c,d = group['box']
            overlap = max(0, min(y1,d)-max(y0,b))
            gap = max(0, x0-c, a-x1)
            if overlap >= .55*min(y1-y0,d-b) and gap <= .8*min(y1-y0,d-b):
                merged.append(i)
        members = [component]
        for i in reversed(merged):
            members.extend(groups.pop(i)['members'])
        points = np.array([m['box'] for m in members])
        box = [int(points[:,0].min()), int(points[:,1].min()),
               int(points[:,2].max()), int(points[:,3].max())]
        groups.append(dict(box=box, members=members))
    result = []
    for group in groups:
        x0,y0,x1,y1 = group['box']
        support = np.isin(labels[y0:y1,x0:x1], [m['label'] for m in group['members']])
        area = int(support.sum())
        if area >= 40 and (x1-x0)*(y1-y0) < mask.size*.15:
            result.append(dict(box=group['box'], support=support,
                               components=[m['box'] for m in group['members']], mass=area))
    return result


def overlap(a, b):
    x0,y0,x1,y1 = a
    u0,v0,u1,v1 = b
    intersection = max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
    area_a, area_b = (x1-x0)*(y1-y0), (u1-u0)*(v1-v0)
    return intersection / max(1, min(area_a,area_b)), intersection/max(1,area_a+area_b-intersection)


def proposals(gray):
    features = np.stack([contrast(frame) >= 18 for frame in gray])
    sums = np.concatenate([np.zeros((1,*features.shape[1:]), np.uint16), np.cumsum(features, axis=0, dtype=np.uint16)])
    found = []
    for lo, hi in temporal_windows(len(gray)):
        stable = (sums[hi]-sums[lo]) >= int(np.ceil((hi-lo)*.8))
        for candidate in boxes_from_mask(stable):
            candidate['windows'] = [[lo,hi]]
            matched = next((i for i,old in enumerate(found) if overlap(old['box'],candidate['box'])[1] >= .55), None)
            if matched is None:
                found.append(candidate)
            else:
                match = found[matched]
                match['windows'].append([lo,hi])
                # Keep the most complete observed repeated shape, not a union
                # of unrelated background pixels from successive windows.
                if candidate['mass'] > match['mass']:
                    candidate['windows'] = match['windows']
                    found[matched] = candidate
    for candidate in found:
        mask = candidate['support']
        x0,y0,x1,y1 = candidate['box']
        coverage = np.mean(features[:,y0:y1,x0:x1][:,mask],axis=1)
        candidate['coverage'] = coverage
        # Occupancy alone confuses changing subtitles with a persistent glyph.
        # Compare signed local structure, and require the same observed shape
        # on multiple independently sampled frames before ranking a proposal.
        possible = np.flatnonzero(coverage >= .8)
        patches = gray[:,y0:y1,x0:x1]
        rows = []
        for patch in patches:
            values = patch.astype(np.float32)
            highpass = cv2.GaussianBlur(values,(3,3),.6)-cv2.GaussianBlur(values,(0,0),3.)
            row = highpass.ravel()
            rows.append(row/max(1e-6,float(np.linalg.norm(row))))
        descriptors = np.stack(rows)
        seeds = possible[np.unique(np.linspace(0,len(possible)-1,min(12,len(possible))).round().astype(int))] if len(possible) else []
        best, anchor = np.zeros(len(gray),np.float32), 0
        for seed in seeds:
            similarities = descriptors @ descriptors[seed]
            if (similarities >= .85).sum() > (best >= .85).sum():
                best,anchor = similarities,int(seed)
        candidate['similarity'] = best
        candidate['shapeAnchor'] = anchor
        candidate['present'] = np.flatnonzero((coverage >= .8)&(best>=.85))
        candidate['score'] = float(np.sqrt(candidate['mass'])*np.log2(1+len(candidate['present'])))
    return sorted(found, key=lambda c:c['score'], reverse=True)


def discover(source, output, budget=96, max_side=1280, limit=40):
    started = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = probe(source)
    gray, indices, scale = sample_source(source, info, output, budget, max_side)
    candidates = proposals(gray)
    reader = Reader(source)
    result = []
    try:
        for item in candidates:
            if len(result) >= limit:
                break
            if len(item['present']) < 3:
                continue
            x0,y0,x1,y1 = item['box']
            original_box = (x0,y0,x1,y1)
            margin = max(3, round(min(x1-x0,y1-y0)*.12))
            x0,y0 = max(0,x0-margin),max(0,y0-margin)
            x1,y1 = min(gray.shape[2],x1+margin),min(gray.shape[1],y1+margin)
            region = np.s_[y0:y1,x0:x1]
            local_support = np.zeros((y1-y0,x1-x0),bool)
            u,v,uu,vv = original_box
            local_support[v-y0:vv-y0,u-x0:uu-x0] = item['support']
            # Prefer a complete repeated shape with a quiet surrounding ring.
            ring = ~cv2.dilate(local_support.astype(np.uint8), np.ones((5,5),np.uint8)).astype(bool)
            scores = []
            for index in item['present']:
                patch = gray[index][region]
                noise = float(np.std(patch[ring])) if ring.any() else 255.
                scores.append((float(item['coverage'][index]) - noise/510., int(index)))
            anchor = max(scores)[1]
            frame_index = int(indices[anchor])
            frame = reader.frame(frame_index)
            box = [max(0,int(np.floor(x0/scale))), max(0,int(np.floor(y0/scale))),
                   min(info['width'],int(np.ceil(x1/scale))), min(info['height'],int(np.ceil(y1/scale)))]
            a,b,c,d = box
            patch = frame[b:d,a:c].copy()
            approximate = cv2.resize(local_support.astype(np.uint8),(c-a,d-b),interpolation=cv2.INTER_NEAREST)>0
            local_contrast = contrast(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY))
            support = (cv2.dilate(approximate.astype(np.uint8),np.ones((3,3),np.uint8))>0)&(local_contrast>=8)
            identifier = f'candidate-{len(result)+1:03}'
            directory = output/identifier
            directory.mkdir(exist_ok=True)
            cv2.imwrite(str(directory/'reference.png'),patch)
            cv2.imwrite(str(directory/'support.png'),support.astype(np.uint8)*255)
            np.save(directory/'support.npy',support,allow_pickle=False)
            observations = [dict(frame=int(frame),coverage=float(coverage),shapeSimilarity=float(similarity),
                                 repeatedContrastObserved=bool(coverage>=.8 and similarity>=.85))
                            for frame,coverage,similarity in zip(indices,item['coverage'],item['similarity'])]
            components = [dict(id=i+1,roi=[int(round(v/scale)) for v in component])
                          for i,component in enumerate(item['components'])]
            result.append(dict(id=identifier,roi=box,referenceFrame=frame_index,
                               supportPath=str(directory/'support.npy'),referencePath=str(directory/'reference.png'),
                               anchorFrames=[int(indices[i]) for i in item['present']],observations=observations,
                               components=components,score=item['score'],supportPixels=int(support.sum()),
                               proposalType='stationary_multiscale_temporal_recurrence',
                               status='candidate_requires_identity',
                               evidence=dict(sampledOccurrences=len(item['present']), temporalWindows=item['windows'],
                                             proposalScale=scale, fullResolutionSupport='observed contrast near repeated proposal')))
            print(json.dumps(dict(stage='candidate',id=identifier,roi=box,occurrences=len(item['present']))),flush=True)
    finally:
        reader.close()
    report = dict(version=1,source=source_signature(source,info),info=info,
                  samplesPath=str(output/'samples.npz'),sampleFrames=indices.tolist(),candidates=result,
                  elapsedSeconds=time.monotonic()-started,
                  limitations=['Proposals do not authorize removal. Repeated scene detail remains ambiguous.',
                               'This initial discovery path handles stationary repeated positions, not arbitrary moving overlays.',
                               'Sparse sampling can miss short events; activity boundaries require further decoding.',
                               'Support is a discovery proposal, not a proven compositing alpha or final repair mask.'])
    (output/'discovery.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=96)
    parser.add_argument('--max-side',type=int,default=1280)
    parser.add_argument('--limit',type=int,default=40)
    args = parser.parse_args()
    if args.samples < 3 or args.max_side < 64 or args.limit < 1:
        parser.error('Discovery requires at least 3 samples, max-side 64, and one candidate')
    discover(args.input,args.output,args.samples,args.max_side,args.limit)


if __name__ == '__main__':
    main()
