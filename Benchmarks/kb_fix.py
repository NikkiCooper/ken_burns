"""Compare current integer-crop approach vs sub-pixel affine warp."""
import cv2, numpy as np
TW, TH = 1920, 1080

def make_src(w, h, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx-w/2)**2 + (yy-h/2)**2) / w
    g = np.clip(127 + 100*np.sin(r*r*900) + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def geom(w, h, direction=True):
    ar = TW/TH
    ww = w; wh = int(w/ar)
    if wh > h: wh = h; ww = int(h*ar)
    wx, wy = (w-ww)//2, (h-wh)//2
    tw_, th_ = int(ww*0.85), int(wh*0.85)
    tx, ty = (w-tw_)//2, int((h-th_)*0.3)
    return ((ww,wh,wx,wy),(tw_,th_,tx,ty)) if direction else ((tw_,th_,tx,ty),(ww,wh,wx,wy))

def track(img, hold, direction, mode):
    h, w = img.shape[:2]
    (w0,h0,x0,y0),(w1,h1,x1,y1) = geom(w, h, direction)
    ar = TW/TH
    out = []
    for f in range(hold):
        ts = (1.0-np.cos((f/max(hold-1,1))*np.pi))/2.0
        cw_f = w0 + (w1-w0)*ts
        ch_f = cw_f/ar
        cx_f = (x0+w0/2.0) + ((x1+w1/2.0)-(x0+w0/2.0))*ts
        cy_f = (y0+h0/2.0) + ((y1+h1/2.0)-(y0+h0/2.0))*ts
        if mode == 'crop':
            icw, ich = round(cw_f), round(ch_f)
            icx, icy = round(cx_f-icw/2.0), round(cy_f-ich/2.0)
            out.append(cv2.resize(img[icy:icy+ich, icx:icx+icw], (TW,TH),
                                  interpolation=cv2.INTER_CUBIC))
        else:  # sub-pixel affine: maps the exact fractional crop rect onto the output
            s = TW / cw_f
            M = np.array([[s, 0.0, TW/2.0 - s*cx_f],
                          [0.0, s, TH/2.0 - s*cy_f]], dtype=np.float64)
            out.append(cv2.warpAffine(img, M, (TW,TH), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE))
    return out

def report(frames, label):
    d = np.array([np.abs(frames[i].astype(np.int16)-frames[i-1].astype(np.int16)).mean()
                  for i in range(1, len(frames))])
    print(f"  {label:<44} dup={int((d==0).sum()):>3}/{len(d)}  "
          f"spike={d.max()/d.mean():.2f}  jerk={np.abs(np.diff(d)).mean()/d.mean():.3f}")

hold = 240
src4k = make_src(3840, 2160)
src1080 = cv2.resize(src4k, (1920,1080), interpolation=cv2.INTER_AREA)

print("\nzoom-in, 240 frames (lower dup/spike/jerk = smoother):")
report(track(src4k,   hold, True, 'crop'),  "4K   src, integer crop  (current)")
report(track(src1080, hold, True, 'crop'),  "1080 src, integer crop  (current, CHOPPY)")
report(track(src4k,   hold, True, 'warp'),  "4K   src, sub-pixel affine warp")
report(track(src1080, hold, True, 'warp'),  "1080 src, sub-pixel affine warp")

print(f"\ncv2.cuda.warpAffine available: {hasattr(cv2.cuda, 'warpAffine')}")
print(f"OpenCV version: {cv2.__version__}")
