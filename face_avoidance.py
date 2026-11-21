"""Face-aware banner placement (FEATURE 1).

The hook banner used to sit at a fixed y (800 for face-tracking
layouts) no matter where the streamer's face was in the 9:16 frame.
These pure functions place the banner — and the quote line — where
they do not cover any face in the rendered clip.

Coordinate systems (read this before touching the mapping):
- The tracker analyses the UNMIRRORED source (e.g. 1920x1080).
- edit_clip mirrors the video with hflip BEFORE the crop when
  mirror=True (the default). The crop window therefore moves in
  mirrored-stream coordinates: a face at unmirrored x=fx is at
  mirrored x=raw_w-fx-fw, and a crop_x computed from the unmirrored
  source must be mirrored too (max_x - crop_x) to land on the same
  face. clip_editor.py does both adjustments and hands this module
  boxes + trajectory in the SAME (rendered-stream) coordinates, so
  the mapping below never has to know about mirroring.
- The 9:16 output is a crop_w x raw_h window of the source scaled to
  1080x1920 with a uniform scale of 1080/crop_w.

No cv2 import here on purpose: the module is pure and unit-testable
under plain python3.
"""

# Output frame geometry (config.VIDEO_WIDTH / VIDEO_HEIGHT).
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# Banner candidate y positions (top edge of the white card) for
# face-tracking layouts. 800 is the historical fixed position and
# stays the default whenever no face is visible.
BANNER_TOP_Y = 260
BANNER_BELOW_FACE_Y = 800
BANNER_BOTTOM_EDGE_Y = 1410  # bottom candidate: box ends at y 1410,
# just above the Kick watermark bar at y 1420.

# Quote line (FEATURE 2) default: text block centred on y 1240.
QUOTE_DEFAULT_Y = 1240
# The Kick watermark bar sits at y 1420; quote text must end above it.
QUOTE_MAX_BOTTOM_Y = 1420

# Safety gaps so text never just touches a face.
FACE_SAFETY_GAP = 20
FACE_REGION_MARGIN = 40


def boxes_overlap(a, b, gap=0):
    """True when boxes a and b overlap, or come within ``gap`` px.

    Boxes are (x1, y1, x2, y2) with x2/y2 exclusive. A gap > 0 treats
    boxes closer than ``gap`` px as overlapping (safety margin).
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 < bx2 + gap and bx1 < ax2 + gap
            and ay1 < by2 + gap and by1 < ay2 + gap)


def overlap_area(a, b):
    """Overlapping area in px^2 (0 when the boxes do not overlap)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    w = min(ax2, bx2) - max(ax1, bx1)
    h = min(ay2, by2) - max(ay1, by1)
    if w <= 0 or h <= 0:
        return 0
    return w * h


def mirror_box(box, raw_w):
    """Mirror a source face box (x, y, w, h) horizontally: the box's
    position in the hflip'd stream. y/w/h are unchanged."""
    fx, fy, fw, fh = box
    return (raw_w - fx - fw, fy, fw, fh)


def mirror_face_boxes(face_boxes, raw_w):
    """Mirror every sampled box of analyze_face_boxes() output
    [(frame_idx, [boxes]), ...] into hflip'd-stream coordinates."""
    return [(frame_idx, [mirror_box(b, raw_w) for b in boxes])
            for frame_idx, boxes in face_boxes]


def mirror_trajectory(smoothed_x, max_x):
    """Mirror a crop_x trajectory so a crop computed from the
    unmirrored source lands on the same faces after hflip:
    mirrored crop_x = max_x - crop_x."""
    return [max_x - x for x in smoothed_x]


def map_box_to_output(box, crop_x, crop_w, raw_h):
    """Map a source face box (x, y, w, h) to 1080x1920 output coords.

    The output is the crop_w x raw_h window starting at crop_x,
    scaled uniformly by 1080/crop_w. The result is clamped to the
    frame; a box fully outside the crop window (or the frame) maps
    to None — it is not visible in the render.
    """
    fx, fy, fw, fh = box
    scale = OUTPUT_WIDTH / crop_w
    x1 = (fx - crop_x) * scale
    y1 = fy * scale
    x2 = (fx + fw - crop_x) * scale
    y2 = (fy + fh) * scale
    # Fully outside the visible window: not in the render at all.
    if x2 <= 0 or y2 <= 0 or x1 >= OUTPUT_WIDTH or y1 >= OUTPUT_HEIGHT:
        return None
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(OUTPUT_WIDTH, int(round(x2)))
    y2 = min(OUTPUT_HEIGHT, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def face_region_for_clip(face_boxes, smoothed_x, crop_w, raw_h,
                         margin=FACE_REGION_MARGIN):
    """Union of every sampled face box mapped to output coords.

    ``face_boxes`` is [(frame_idx, [(x, y, w, h), ...]), ...] in the
    same coordinate system as ``smoothed_x`` (clip_editor mirrors
    both first when the render is mirrored). Each sample uses the
    crop_x of its own frame. Returns (x1, y1, x2, y2) in output
    coords, expanded by ``margin`` px on every side, or None when
    no face was visible in any sample.
    """
    if not face_boxes or not smoothed_x:
        return None
    n_frames = len(smoothed_x)
    x1 = y1 = None
    x2 = y2 = None
    for frame_idx, boxes in face_boxes:
        idx = max(0, min(n_frames - 1, int(frame_idx)))
        crop_x = smoothed_x[idx]
        for box in boxes:
            mapped = map_box_to_output(box, crop_x, crop_w, raw_h)
            if mapped is None:
                continue
            mx1, my1, mx2, my2 = mapped
            x1 = mx1 if x1 is None else min(x1, mx1)
            y1 = my1 if y1 is None else min(y1, my1)
            x2 = mx2 if x2 is None else max(x2, mx2)
            y2 = my2 if y2 is None else max(y2, my2)
    if x1 is None:
        return None
    return (max(0, x1 - margin),
            max(0, y1 - margin),
            min(OUTPUT_WIDTH, x2 + margin),
            min(OUTPUT_HEIGHT, y2 + margin))


def _banner_box(y, box_h, box_w):
    """The banner card's output box for a top-edge y: centred
    horizontally, ``box_w`` wide, ``box_h`` tall."""
    x1 = (OUTPUT_WIDTH - box_w) // 2
    return (x1, y, x1 + box_w, y + box_h)


def choose_banner_y(face_region, box_h, layout, box_w=None):
    """Pick the banner's top-edge y for a face-tracking layout.

    Candidates, in order: top (260), below-face (800 — the current
    behaviour), bottom (1410 - box_h, just above the Kick bar). The
    first candidate whose box does not overlap the face region
    (with a 20 px safety gap) wins. When every candidate overlaps,
    the one with the smallest overlap area wins. With no face
    region, 800 is returned — the current behaviour, no regression.

    ``layout`` is accepted for forward compatibility (non-face-
    tracking layouts never call this — they keep their fixed y)
    and is currently unused. ``box_w`` is the real card width when
    known, else 800 (the card is at most ~790 px wide incl. margins).
    """
    if face_region is None:
        return BANNER_BELOW_FACE_Y
    width = box_w if box_w else 800
    candidates = [BANNER_TOP_Y,
                  BANNER_BELOW_FACE_Y,
                  BANNER_BOTTOM_EDGE_Y - box_h]
    best_y = None
    best_overlap = None
    for y in candidates:
        box = _banner_box(y, box_h, width)
        if not boxes_overlap(box, face_region, gap=FACE_SAFETY_GAP):
            return y
        area = overlap_area(box, face_region)
        if best_overlap is None or area < best_overlap:
            best_overlap = area
            best_y = y
    return best_y


def choose_quote_y(face_region, banner_box, block_h):
    """Pick the quote line's centre y (default 1240).

    Candidates stay in the lower third, above the Kick bar (the
    text block must end above y 1420) and below the banner box.
    The current centre y 1240 is tried first; then positions that
    step down toward the Kick bar, then up. The first candidate
    whose block overlaps neither the face region nor the banner box
    (20 px safety gap) wins; otherwise the least-overlapping one.
    With no face region, 1240 is returned.
    """
    if face_region is None:
        return QUOTE_DEFAULT_Y
    # Candidate centre positions: current first, then below (towards
    # the Kick bar, stopping so the block ends above y 1420), then
    # above (staying below the banner box).
    max_centre = QUOTE_MAX_BOTTOM_Y - block_h // 2 - 10
    candidates = [QUOTE_DEFAULT_Y]
    step = 60
    y = QUOTE_DEFAULT_Y + step
    while y <= max_centre:
        candidates.append(y)
        y += step
    # Above the default: only positions whose block clears the
    # banner box's bottom edge, when the banner sits above the
    # quote. A bottom banner does not bound the upward search —
    # the overlap check below still keeps the quote off it.
    if banner_box and banner_box[3] <= QUOTE_DEFAULT_Y:
        min_centre = banner_box[3] + block_h // 2 + FACE_SAFETY_GAP
    else:
        min_centre = block_h // 2 + FACE_SAFETY_GAP
    y = QUOTE_DEFAULT_Y - step
    while y >= min_centre and y > 0:
        candidates.append(y)
        y -= step
    best_y = None
    best_overlap = None
    for centre_y in candidates:
        top = centre_y - block_h // 2
        # Quote text spans most of the frame width (max 900 px).
        box = ((OUTPUT_WIDTH - 900) // 2, top,
               (OUTPUT_WIDTH + 900) // 2, top + block_h)
        overlaps = boxes_overlap(box, face_region, gap=FACE_SAFETY_GAP)
        total = overlap_area(box, face_region)
        if banner_box:
            overlaps = overlaps or boxes_overlap(
                box, banner_box, gap=FACE_SAFETY_GAP)
            total += overlap_area(box, banner_box)
        if not overlaps:
            return centre_y
        if best_overlap is None or total < best_overlap:
            best_overlap, best_y = total, centre_y
    return best_y if best_y is not None else QUOTE_DEFAULT_Y
