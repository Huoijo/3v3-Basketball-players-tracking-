"""Isolated GPU tracking probe. Embedded in the notebook and run in a subprocess.

No probe tracker state or observations enter the production baseline.
"""
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from ultralytics import YOLO

# ORIGINAL_PARSER


def check_result_label(parts, box):
    """Compare the original parser with the actual tracked Result object."""
    parsed = parse_yolo_tracking_line(parts)
    expected = (int(box.id.item()), float(box.conf.item()), *box.xywhn.reshape(-1).tolist())
    if len(parts) != 7 or parsed[0] != expected[0] or not all(
        math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6) for a, b in zip(parsed[1:], expected[1:])
    ):
        raise RuntimeError(f'Original parser does not match actual tracked labels: line={parts}, parsed={parsed}, expected={expected}. Inspect evidence; no speculative parser patch was applied.')
    return {'line': ' '.join(parts), 'parsed': list(parsed), 'expected_from_result': list(expected)}


def main():
    video, weights, run_dir, max_frames = sys.argv[1:]
    run_dir = Path(run_dir)
    destination = run_dir / 'label_probe'
    destination.mkdir(exist_ok=False)
    summary = dict(status='RUNNING', frames=0, tracked_label_rows=0, examples=[], parser_changed=False)
    try:
        model = YOLO(weights)
        results = model.track(source=video, tracker=str(run_dir / 'bytetrack.yaml'), device='0',
                              classes=[0], save=False, save_txt=True, save_conf=True,
                              stream=True, verbose=False, project=str(destination), name='tracking', exist_ok=False)
        previous = set()
        try:
            for index, result in enumerate(results, start=1):
                summary['frames'] = index
                labels = set((Path(result.save_dir) / 'labels').glob('*.txt'))
                new_labels = labels - previous
                previous = labels
                boxes = result.boxes
                if boxes is not None and len(boxes):
                    if not boxes.is_track:
                        raise RuntimeError('Actual output contains untracked detection-only boxes. The original six-field parser has not been verified for this format.')
                    if len(new_labels) != 1:
                        raise RuntimeError(f'Expected one actual saved label file at sample frame {index}; found {new_labels}')
                    label = next(iter(new_labels))
                    lines = label.read_text().splitlines()
                    if len(lines) != len(boxes):
                        raise RuntimeError(f'Saved label count differs from Results boxes in {label}')
                    for line, box in zip(lines, boxes):
                        item = check_result_label(line.split(), box)
                        summary['tracked_label_rows'] += 1
                        if len(summary['examples']) < 5:
                            summary['examples'].append(dict(frame=index, file=str(label), **item))
                if index >= int(max_frames):
                    break
        finally:
            results.close()
        if not summary['tracked_label_rows']:
            raise RuntimeError('No tracked label rows in real sample; parser compatibility is unverified. Inspect the original video and runtime.')
        summary['status'] = 'PASS'
    except BaseException:
        summary['status'] = 'FAIL'
        summary['error'] = traceback.format_exc()
        raise
    finally:
        (destination / 'label_format.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
