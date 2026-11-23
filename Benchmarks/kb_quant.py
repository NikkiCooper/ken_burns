import numpy as np

TARGET_W, TARGET_H = 1920, 1080

def slide_params(w, h, drift_value=1.0, direction=True):
    ar = TARGET_W / TARGET_H
    wide_w = w
    wide_h = int(w / ar)
    if wide_h > h:
        wide_h = h
        wide_w = int(h * ar)
    wide_x = (w - wide_w) // 2
    wide_y = (h - wide_h) // 2
    scale = 1.0 - drift_value * 0.15
    tight_w = int(wide_w * scale)
    tight_h = int(wide_h * scale)
    tight_x = (w - tight_w) // 2
    tight_y = int((h - tight_h) * 0.3)
    if direction:
        return (wide_w, wide_h, wide_x, wide_y), (tight_w, tight_h, tight_x, tight_y)
    return (tight_w, tight_h, tight_x, tight_y), (wide_w, wide_h, wide_x, wide_y)

def analyse(w, h, hold_frames, direction=True, label=""):
    start, end = slide_params(w, h, 1.0, direction)
    w0, h0, x0, y0 = start
    w1, h1, x1, y1 = end
    ar = TARGET_W / TARGET_H
    rows = []
    for f in range(hold_frames):
        t = f / max(hold_frames - 1, 1)
        t_s = (1.0 - np.cos(t * np.pi)) / 2.0
        cw_f = w0 + (w1 - w0) * t_s
        ch_f = cw_f / ar
        cx_f = (x0 + w0/2.0) + ((x1 + w1/2.0) - (x0 + w0/2.0)) * t_s
        cy_f = (y0 + h0/2.0) + ((y1 + h1/2.0) - (y0 + h0/2.0)) * t_s
        icw, ich = round(cw_f), round(ch_f)
        icx, icy = round(cx_f - icw/2.0), round(cy_f - ich/2.0)
        # ideal (unquantized) output-space mapping of the crop's top-left corner
        # magnification = TARGET_W / crop_width
        rows.append((f, cw_f, cy_f, icw, ich, icx, icy))

    # duplicate-frame runs: identical (icw,ich,icx,icy) => bit-identical output
    runs = []
    prev = None
    for f, cw_f, cy_f, icw, ich, icx, icy in rows:
        key = (icw, ich, icx, icy)
        if key == prev:
            runs[-1][1] += 1
        else:
            runs.append([f, 1, key])
            prev = key

    # output-space jump when geometry does change
    jumps = []
    for i in range(1, len(runs)):
        pw, ph, px, py = runs[i-1][2]
        cw, ch, cx, cy = runs[i][2]
        mag = TARGET_W / cw
        # vertical displacement of image content, in OUTPUT pixels
        dy_out = abs(cy - py) * mag
        # scale jump expressed as edge displacement in output px
        dscale_out = abs(cw - pw) / 2.0 * mag
        jumps.append(dy_out + dscale_out)

    dup = [r[1] for r in runs]
    print(f"\n=== {label}  src {w}x{h}  {'zoom-in' if direction else 'zoom-out'} ===")
    print(f"  wide crop   : {w0 if direction else w1}x{h0 if direction else h1}")
    print(f"  tight crop  : {w1 if direction else w0}x{h1 if direction else h0}")
    print(f"  magnification range : {TARGET_W/max(w0,w1):.4f}x -> {TARGET_W/min(w0,w1):.4f}x")
    print(f"  distinct geometries : {len(runs)} over {hold_frames} frames")
    print(f"  longest frozen run  : {max(dup)} frames ({max(dup)/30:.2f}s at 30fps)")
    print(f"  frozen runs >= 3 frm: {sum(1 for d in dup if d >= 3)}")
    print(f"  mean/max jump when it moves (output px): {np.mean(jumps):.3f} / {np.max(jumps):.3f}")
    print(f"  first 12 run lengths : {dup[:12]}")
    print(f"  first 8 geometries   : {[r[2] for r in runs[:8]]}")

hold = int(8.0 * 30)
for d in (True, False):
    analyse(3840, 2160, hold, d, "4K source")
    analyse(1920, 1080, hold, d, "1080p source")
