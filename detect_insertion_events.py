#!/usr/bin/env python3
"""
Detect insertion events in TIRF microscopy image stacks.

This script processes a time series of TIRF images and identifies
the appearance of new fluorescent spots (receptor insertions).
Outputs a CSV with event coordinates, frame numbers, and optionally
amplitude metrics.

Author: Multi-lab Consortium
"""

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
import argparse
import os
import sys

def load_tiff_stack(filepath):
    """Load a TIFF stack as a 3D numpy array (frames, height, width)."""
    with tifffile.TiffFile(filepath) as tif:
        stack = tif.asarray()
    return stack

def detect_insertions_slow(stack, threshold=5.0, min_distance=3):
    """
    Naive frame-by-frame differencing detection.
    
    Parameters
    ----------
    stack : ndarray (T, Y, X)
        Image stack.
    threshold : float
        Intensity threshold for detection (multiples of std).
    min_distance : int
        Minimum pixel distance between distinct events.
    
    Returns
    -------
    events : list of dict
        Each dict contains 'frame', 'y', 'x', 'intensity'.
    """
    events = []
    T, Y, X = stack.shape
    # Compute background statistics from first frame
    bg_mean = np.mean(stack[0])
    bg_std = np.std(stack[0])
    for t in range(1, T):
        diff = stack[t] - stack[t-1]
        # Simple thresholding
        candidates = diff > threshold * bg_std
        labeled, num_features = ndimage.label(candidates)
        for i in range(1, num_features+1):
            mask = labeled == i
            y, x = np.where(mask)
            if len(y) == 0:
                continue
            # centroid
            cy = int(np.mean(y))
            cx = int(np.mean(x))
            # ensure not too close to previous events in same frame
            # (simplistic)
            events.append({
                'frame': t,
                'y': cy,
                'x': cx,
                'intensity': diff[cy, cx]
            })
    return events

def save_events_to_csv(events, output_path):
    """Save events list to CSV."""
    df = pd.DataFrame(events)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} events to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Detect insertion events in TIRF stack.')
    parser.add_argument('input', help='Path to input TIFF stack')
    parser.add_argument('--output', default='events.csv', help='Output CSV path')
    parser.add_argument('--threshold', type=float, default=5.0, help='Threshold (standard deviations)')
    parser.add_argument('--min_distance', type=int, default=3, help='Minimum pixel distance between events')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        sys.exit(f"Input file not found: {args.input}")
    
    print(f"Loading stack from {args.input}...")
    stack = load_tiff_stack(args.input)
    print(f"Stack shape: {stack.shape}")
    
    print("Detecting insertion events (slow method)...")
    events = detect_insertions_slow(stack, threshold=args.threshold, min_distance=args.min_distance)
    
    print(f"Detected {len(events)} events.")
    save_events_to_csv(events, args.output)

if __name__ == '__main__':
    main()