"""Measure frame-to-frame motion smoothness for 4K vs 1080p sources,
using the exact geometry from get_gpu_frame() + INTER_CUBIC resize."""
import cv2, numpy as np

TW, TH = 1920, 1080

def make_src(w, h, seed=0):
    """Detailed synthetic image: radial chirp + texture (worst case for judder)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w/2, h/2
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2) / w
    chirp = 127 + 100*np.sin(r*r*900)
    tex = rng.normal(0, 8, (h, w))
    g = np.clip(chirp + tex, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def geom(w, h, direction=True):
    ar = TW/TH
    ww = w; wh = int(w/ar)
    if wh > h: wh = h; ww = int(h*ar)
    wx, wy = (w-ww)//2, (h-wh)//2
    s = 0.85
    tw_, th_ = int(ww*s), int(wh*s)
    tx, ty = (w-tw_)//2, int((h-th_)*0.3)
    return ((ww,wh,wx,wy),(tw_,th_,tx,ty)) if direction else ((tw_,th_,tx,ty),(ww,wh,wx,wy))

def render(img, hold, direction=True):
    h, w = img.shape[:2]
    (w0,h0,x0,y0),(w1,h1,x1,y1) = geom(w, h, direction)
    ar = TW/TH
    out = []
    for f in range(hold):
        t = f/max(hold-1,1)
        ts = (1.0-np.cos(t*np.pi))/2.0
        cw_f = w0 + (w1-w0)*ts
        ch_f = cw_f/ar
        cx_f = (x0+w0/2.0) + ((x1+w1/2.0)-(x0+w0/2.0))*ts
        cy_f = (y0+h0/2.0) + ((y1+h1/2.0)-(y0+h0/2.0))*ts
        icw, ich = round(cw_f), round(ch_f)
        icx, icy = round(cx_f-icw/2.0), round(cy_f-ich/2.0)
        crop = img[icy:icy+ich, icx:icx+icw]
        out.append(cv2.resize(crop, (TW,TH), interpolation=cv2.INTER_CUBIC))
    return out

def report(frames, label):
    d = np.array([np.abs(frames[i].astype(np.int16)-frames[i-1].astype(np.int16)).mean()
                  for i in range(1, len(frames))])
    zeros = int((d == 0).sum())
    nz = d[d > 0]
    # jerk: how erratic is the per-frame change (0 = perfectly even motion)
    jerk = np.abs(np.diff(d)).mean() / d.mean()
    print(f"\n--- {label} ---")
    print(f"  identical consecutive frames : {zeros} / {len(d)}")
    print(f"  mean |Δ| per frame           : {d.mean():.3f}")
    print(f"  max  |Δ| per frame           : {d.max():.3f}")
    print(f"  max/mean ratio (spikiness)   : {d.max()/d.mean():.2f}")
    print(f"  normalised jerk              : {jerk:.3f}   <-- lower = smoother")
    print(f"  first 20 per-frame |Δ|       : {np.round(d[:20],2).tolist()}")

hold = 240
src4k = make_src(3840, 2160)
src1080 = cv2.resize(src4k, (1920,1080), interpolation=cv2.INTER_AREA)

report(render(src4k, hold, True),   "4K source 3840x2160 -> 1080p out (zoom-in)")
report(render(src1080, hold, True), "1080p source 1920x1080 -> 1080p out (zoom-in)")

# Is the wide-end resize a literal pass-through at 1:1?
crop = src1080[0:1080, 0:1920]
rs = cv2.resize(crop, (TW,TH), interpolation=cv2.INTER_CUBIC)
print(f"\n1:1 INTER_CUBIC resize is bit-identical to input: {np.array_equal(crop, rs)}")
