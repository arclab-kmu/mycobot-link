#!/usr/bin/env python3
"""RealSense D435i web viewer over raw V4L2 (no librealsense needed).

Color  : /dev/video2  YUYV 640x480
Depth  : /dev/video0  Z16  640x480 (uint16, millimetres; 0 = no depth)

Frames come from `v4l2-ctl --stream-to=-` subprocesses; the server turns them
into MJPEG streams plus a small page. It binds to 127.0.0.1 by default, so view
it through an SSH tunnel:  ssh -p 22022 -N -L 8090:127.0.0.1:8090 er@<jetson>

Endpoints: /  /color.mjpg  /depth.mjpg  /color.jpg  /depth.jpg  /depth.png
           /depth_at?x=&y=  /stats  /set?near=&far=   (near/far in mm)
Color and depth are NOT aligned (different sensors and FOV).
"""
import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

log = logging.getLogger("rs_web_viewer")
STOP = threading.Event()
CHILDREN = []


class V4L2Stream(threading.Thread):
    """Keeps the latest raw frame of one V4L2 node; restarts v4l2-ctl on failure."""

    def __init__(self, name, dev, pixfmt, width, height, fps, bytes_per_px):
        super().__init__(daemon=True, name=name)
        self.dev, self.pixfmt = dev, pixfmt
        self.w, self.h, self.fps = width, height, fps
        self.size = width * height * bytes_per_px
        self.cond = threading.Condition()
        self.raw, self.seq, self.t = None, 0, 0.0
        self.times = deque(maxlen=30)

    def _read_exact(self, f):
        buf = bytearray()
        while len(buf) < self.size:
            chunk = f.read(self.size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def run(self):
        while not STOP.is_set():
            subprocess.run(
                ["v4l2-ctl", "-d", self.dev,
                 "--set-fmt-video=width=%d,height=%d,pixelformat=%s" % (self.w, self.h, self.pixfmt),
                 "--set-parm=%d" % self.fps],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p = subprocess.Popen(
                ["v4l2-ctl", "-d", self.dev, "--stream-mmap", "--stream-to=-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            CHILDREN.append(p)
            log.info("%s: streaming %s %s %dx%d@%d", self.name, self.dev, self.pixfmt, self.w, self.h, self.fps)
            while not STOP.is_set():
                raw = self._read_exact(p.stdout)
                if raw is None:
                    break
                now = time.time()
                with self.cond:
                    self.raw, self.t = raw, now
                    self.seq += 1
                    self.times.append(now)
                    self.cond.notify_all()
            p.kill()
            p.wait()
            if not STOP.is_set():
                log.warning("%s: stream ended, retrying in 2 s", self.name)
                time.sleep(2)

    def wait_newer(self, seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq or STOP.is_set(), timeout)
            return self.seq, self.raw

    def stats(self):
        with self.cond:
            ts = list(self.times)
            fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
            age = time.time() - self.t if self.t else None
        return {"fps": round(fps, 1), "age_s": None if age is None else round(age, 2), "frames": self.seq}


class Viewer:
    def __init__(self, color, depth, near_mm, far_mm, quality):
        self.color, self.depth = color, depth
        self.near, self.far = near_mm, far_mm
        self.quality = quality
        self.lock = threading.Lock()
        self.cache = {}  # key -> (cache_key, jpeg)

    def depth_array(self, raw):
        return np.frombuffer(raw, dtype="<u2").reshape(self.depth.h, self.depth.w)

    def render(self, which, raw):
        if which == "color":
            yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(self.color.h, self.color.w, 2)
            img = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        else:
            d = self.depth_array(raw).astype(np.float32)
            near, far = self.near, self.far
            norm = np.clip((d - near) / max(far - near, 1.0), 0.0, 1.0)
            img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            img[d == 0] = 0
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return jpg.tobytes() if ok else None

    def jpeg(self, which, last_seq=-1, timeout=2.0):
        stream = self.color if which == "color" else self.depth
        seq, raw = stream.wait_newer(last_seq, timeout)
        if raw is None:
            return seq, None
        key = (seq, self.near, self.far) if which == "depth" else (seq,)
        with self.lock:
            hit = self.cache.get(which)
            if hit and hit[0] == key:
                return seq, hit[1]
        jpg = self.render(which, raw)
        with self.lock:
            self.cache[which] = (key, jpg)
        return seq, jpg


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RealSense Viewer</title>
<style>
:root{--fg:#111;--muted:#666;--line:#e5e5e5;--accent:#0f9fa8;--bg:#fff}
@media (prefers-color-scheme:dark){:root{--fg:#eee;--muted:#999;--line:#333;--bg:#111}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
 font:14px/1.5 system-ui,-apple-system,"Noto Sans KR",sans-serif}
h1{font-size:16px;font-weight:600;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 14px}
.row{display:flex;flex-wrap:wrap;gap:16px}
figure{margin:0;flex:1 1 420px;min-width:0;position:relative}
figure img{width:100%;display:block;border:1px solid var(--line);background:#000}
figcaption{display:flex;justify-content:space-between;gap:8px;color:var(--muted);margin-top:6px}
#tip{position:absolute;pointer-events:none;background:rgba(0,0,0,.75);color:#fff;
 padding:2px 6px;border-radius:3px;font-size:12px;display:none;white-space:nowrap}
.bar{display:flex;align-items:center;gap:8px;margin-top:6px}
.grad{flex:1;height:8px;border:1px solid var(--line);
 background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}
.ctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:14px;padding-top:12px;
 border-top:1px solid var(--line)}
input[type=number]{width:72px;padding:3px 6px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
button{padding:4px 10px;border:1px solid var(--accent);background:none;color:var(--accent);cursor:pointer}
a{color:var(--accent)}
#stats{color:var(--muted);font-variant-numeric:tabular-nums}
</style></head><body>
<h1>RealSense D435i</h1>
<p class="sub">컬러와 depth는 정렬되지 않은 원본입니다(센서와 시야각이 다름). depth 위에 마우스를 올리면 거리를 보여 줍니다.</p>
<div class="row">
 <figure><img id="c" src="color.mjpg" alt="color stream">
  <figcaption><span>color · /dev/video2</span><a href="color.jpg" download>snapshot</a></figcaption></figure>
 <figure id="df"><img id="d" src="depth.mjpg" alt="depth stream"><div id="tip"></div>
  <figcaption><span>depth · /dev/video0 · 검정 = 값 없음</span>
   <span><a href="depth.jpg" download>snapshot</a> · <a href="depth.png" download>raw 16-bit (mm)</a></span></figcaption>
  <div class="bar"><span id="nl"></span><div class="grad"></div><span id="fl"></span></div></figure>
</div>
<div class="ctl">
 <label>near <input id="near" type="number" step="0.05" min="0"> m</label>
 <label>far <input id="far" type="number" step="0.05" min="0.1"> m</label>
 <button id="apply">범위 적용</button>
 <span id="stats"></span>
</div>
<script>
const $=id=>document.getElementById(id);
async function stats(){
 try{const s=await (await fetch('stats')).json();
  const f=k=>`${k} ${s[k].fps} fps`+(s[k].age_s==null?'':` (${s[k].age_s}s)`);
  $('stats').textContent=f('color')+' · '+f('depth');
  if(document.activeElement!==$('near')&&document.activeElement!==$('far')){
   $('near').value=(s.near_mm/1000).toFixed(2);$('far').value=(s.far_mm/1000).toFixed(2);}
  $('nl').textContent=(s.near_mm/1000).toFixed(2)+' m';$('fl').textContent=(s.far_mm/1000).toFixed(2)+' m';
 }catch(e){$('stats').textContent='서버 응답 없음';}
}
setInterval(stats,1000);stats();
$('apply').onclick=async()=>{
 const n=Math.round(parseFloat($('near').value)*1000),f=Math.round(parseFloat($('far').value)*1000);
 await fetch(`set?near=${n}&far=${f}`);stats();};
let busy=false;
$('d').addEventListener('mousemove',async ev=>{
 if(busy)return;busy=true;
 const img=$('d'),r=img.getBoundingClientRect();
 const x=Math.floor((ev.clientX-r.left)/r.width*640),y=Math.floor((ev.clientY-r.top)/r.height*480);
 try{const j=await (await fetch(`depth_at?x=${x}&y=${y}`)).json();
  const t=$('tip');t.style.display='block';
  t.style.left=(ev.clientX-r.left+12)+'px';t.style.top=(ev.clientY-r.top+12)+'px';
  t.textContent=j.mm?`${(j.mm/1000).toFixed(3)} m  (${x}, ${y})`:`값 없음  (${x}, ${y})`;
 }catch(e){}
 setTimeout(()=>busy=false,80);
});
$('d').addEventListener('mouseleave',()=>{$('tip').style.display='none';});
</script></body></html>
"""


def make_handler(viewer, max_fps):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(200, "application/json", json.dumps(obj).encode())

        def _mjpeg(self, which, fps):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            period, last, t_last = 1.0 / max(fps, 0.5), -1, 0.0
            try:
                while not STOP.is_set():
                    seq, jpg = viewer.jpeg(which, last)
                    if jpg is None or seq == last:
                        continue
                    last = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg + b"\r\n")
                    wait = period - (time.time() - t_last)
                    if wait > 0:
                        time.sleep(wait)
                    t_last = time.time()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = u.path.rstrip("/") or "/"
            if path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif path in ("/color.mjpg", "/depth.mjpg"):
                fps = float(q.get("fps", [max_fps])[0])
                self._mjpeg(path[1:6], min(fps, max_fps))
            elif path in ("/color.jpg", "/depth.jpg"):
                _, jpg = viewer.jpeg(path[1:6])
                if jpg is None:
                    self._send(503, "text/plain", b"no frame yet")
                else:
                    self._send(200, "image/jpeg", jpg)
            elif path == "/depth.png":
                raw = viewer.depth.raw
                if raw is None:
                    self._send(503, "text/plain", b"no frame yet")
                    return
                ok, png = cv2.imencode(".png", viewer.depth_array(raw))
                self._send(200, "image/png", png.tobytes())
            elif path == "/depth_at":
                raw = viewer.depth.raw
                try:
                    x, y = int(q["x"][0]), int(q["y"][0])
                except (KeyError, ValueError):
                    self._send(400, "text/plain", b"need x and y")
                    return
                mm = None
                if raw is not None and 0 <= x < viewer.depth.w and 0 <= y < viewer.depth.h:
                    d = viewer.depth_array(raw)
                    win = d[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
                    v = win[win > 0]
                    mm = int(np.median(v)) if v.size else None
                self._json({"x": x, "y": y, "mm": mm})
            elif path == "/stats":
                self._json({"color": viewer.color.stats(), "depth": viewer.depth.stats(),
                            "near_mm": viewer.near, "far_mm": viewer.far})
            elif path == "/set":
                try:
                    near = float(q.get("near", [viewer.near])[0])
                    far = float(q.get("far", [viewer.far])[0])
                except ValueError:
                    self._send(400, "text/plain", b"bad near/far")
                    return
                if 0 <= near < far:
                    viewer.near, viewer.far = near, far
                self._json({"near_mm": viewer.near, "far_mm": viewer.far})
            else:
                self._send(404, "text/plain", b"not found")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--color-dev", default="/dev/video2")
    ap.add_argument("--depth-dev", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15, help="camera fps for both nodes")
    ap.add_argument("--max-stream-fps", type=float, default=15.0)
    ap.add_argument("--near", type=float, default=0.20, help="depth colormap near [m]")
    ap.add_argument("--far", type=float, default=1.50, help="depth colormap far [m]")
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    color = V4L2Stream("color", args.color_dev, "YUYV", args.width, args.height, args.fps, 2)
    depth = V4L2Stream("depth", args.depth_dev, "Z16 ", args.width, args.height, args.fps, 2)
    viewer = Viewer(color, depth, args.near * 1000, args.far * 1000, args.quality)
    color.start()
    depth.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(viewer, args.max_stream_fps))
    server.daemon_threads = True

    def shutdown(signum, frame):
        STOP.set()
        for p in CHILDREN:
            if p.poll() is None:
                p.kill()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log.info("viewer on http://%s:%d", args.host, args.port)
    server.serve_forever()
    log.info("stopped")
    sys.exit(0)


if __name__ == "__main__":
    main()
