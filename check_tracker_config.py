"""
Run this from your project root:  python check_tracker_config.py

Shows exactly what confidence/iou/imgsz values the tracker is actually
using right now, and confirms the custom ByteTrack yaml was generated
with the right thresholds.
"""
import sys
sys.path.insert(0, '.')

from config import Config
from src.tracker import VehicleTracker

cfg = Config()
vt = VehicleTracker(cfg)
vt.init_direction('north')

print('--- Detection config actually in use ---')
print('confidence:', cfg.detection.confidence)
print('iou_threshold:', cfg.detection.iou_threshold)
print('imgsz:', cfg.detection.imgsz)

print()
print('--- ByteTrack actually loaded for north? ---')
print(vt._use_bytetrack.get('north'))

print()
print('--- Tracker yaml path ---')
yaml_path = getattr(vt, '_tracker_yaml_path', None)
print(yaml_path if yaml_path else 'NOT SET — this is a bug, it should exist')

import os
if yaml_path and os.path.exists(yaml_path):
    print()
    print('--- Tracker yaml content ---')
    print(open(yaml_path).read())
elif yaml_path:
    print('Path was set but file does not exist on disk:', yaml_path)
