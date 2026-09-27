"""═══ المؤثرات الصوتية — مكتبة + جدول مستويات ═══
   python3 05_sfx.py <work>            → يبني sfx.wav من sfx.json
   python3 05_sfx.py <work> --list     → يطبع الأنواع والمستويات ومصدر كل صوت (ملف أو مولَّد)

sfx.json — كل نوع قائمة أوقات بالثواني (نفس الشكل القديم، فـ10_script_edit يزيحها عادي):
{ "outro": 5.2,
  "whoosh": [3.1, 11.25], "riser": [6.4], "impact": [7.85], "stamp": [9.1],
  "click": [2.0, 2.4], "tick": [12.0, 12.1, 12.2], "keys": [14.0],
  "register": [20.3], "thin": [22.0], "shutter": [25.5],
  "levels": { "tick": -24 },            ← اختياري: غيّر مستوى نوع (dB)
  "gain": 0 }                           ← اختياري: رفع/خفض كل المؤثرات مرة وحدة (dB)

المصدر: لو فيه ملف صوت حقيقي بإسم النوع يُستخدم هو، وإلا يتولّد بالكود.
  البحث بالترتيب: <work>/sfx-lib/<type>.*  ثم  <skill>/assets/sfx/<type>.*
  (wav · mp3 · m4a · aac · aif — أي ملف ffmpeg يقراه)
"""
import sys, os, json, glob, subprocess
import numpy as np

S = os.path.abspath(sys.argv[1]) + "/"
LIST = "--list" in sys.argv
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000
rng = np.random.RandomState(11)

# ── جدول المستويات (ذروة الصوت بالـdBFS) ────────────────────────────────
# الفكرة: كل حركة على الشاشة ليها صوت، بس الخلفية واطية جداً والضربات قليلة وواضحة.
#   خلفية خفيفة   −27…−20   عداد · كيبورد · كوين
#   حركات          −16…−12   كرت ينوّر · قلم يشطب · نقرة
#   ضربة واحدة     −12…−10   ختم · مطرقة · impact — للحظة المهمة بس
#   ووش على القطع  ≈ −26     بس لما القطع ينقل لجرافيك جديد
LEVELS = {
    # خلفية
    "tick": -24, "keys": -24, "coin": -22,
    # حركات
    "click": -15, "pop": -15, "pen": -15, "card": -16, "tap": -16,
    "thin": -15, "register": -14, "shutter": -16,
    # ضربات
    "impact": -11, "stamp": -10, "thud": -11, "hammer": -10,
    # انتقالات
    "whoosh": -26, "whoosh_up": -26, "whoosh_down": -26, "riser": -19,
}
HITS = {"impact", "stamp", "thud", "hammer"}   # هذي بس اللي تنحسب بحد الـ15 بالدقيقة


# ── مولّدات الأصوات (لو ما فيه ملف حقيقي) ────────────────────────────────
def lp(x, a0, a1):
    y = np.empty_like(x); z = 0.0
    for i in range(len(x)):
        a = a0 + (a1 - a0) * (i / len(x)); z += a * (x[i] - z); y[i] = z
    return y

def T(d): return np.arange(int(d * SR)) / SR
def norm(s): return s / (np.max(np.abs(s)) + 1e-9)

def g_whoosh(up=True, d=0.34):
    t = T(d); y = lp(rng.randn(len(t)), 0.03, 0.30) if up else lp(rng.randn(len(t)), 0.30, 0.03)
    return norm(y) * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.6

def g_thud(f0=135, f1=58, d=0.30):
    t = T(d); f = f0 * np.exp(np.log(f1 / f0) * t / d)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.085)
    s += lp(rng.randn(len(t)), 0.35, 0.05) * np.exp(-t / 0.006) * 0.35
    return norm(s)

def g_tap(d=0.09):
    t = T(d); s = lp(rng.randn(len(t)), 0.22, 0.05) * np.exp(-t / 0.013)
    return norm(s * np.minimum(1.0, t / 0.0012))

def g_click(d=0.05):
    t = T(d); s = np.sin(2 * np.pi * 2400 * t) * np.exp(-t / 0.004)
    s += lp(rng.randn(len(t)), 0.6, 0.2) * np.exp(-t / 0.002) * 0.6
    return norm(s)

def g_tick(d=0.03):
    t = T(d); return norm(np.sin(2 * np.pi * 3200 * t) * np.exp(-t / 0.0025))

def g_keys(d=0.45):                      # رشقة كيبورد: ٦ نقرات عشوائية
    t = T(d); s = np.zeros(len(t)); k = g_tap(0.04)
    for p in sorted(rng.uniform(0, d - 0.05, 6)):
        i = int(p * SR); s[i:i + len(k)] += k[:len(s) - i] * rng.uniform(0.6, 1.0)
    return norm(s)

def g_pop(d=0.12):
    t = T(d); f = 900 * np.exp(-t / 0.03) + 300
    return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.03))

def g_pen(d=0.35):                        # شطب قلم: خرخشة بتتحرك
    t = T(d); s = lp(rng.randn(len(t)), 0.5, 0.35)
    return norm(s * np.sin(np.pi * t / d) ** 0.7 * (1 + 0.5 * np.sin(2 * np.pi * 18 * t)))

def g_card(d=0.22):                        # كرت بينوّر: ووش صغير + بريق
    t = T(d); s = g_whoosh(True, d) * 0.6 + np.sin(2 * np.pi * 1760 * t) * np.exp(-t / 0.06) * 0.4
    return norm(s)

def g_thin(d=0.7):                         # إجابة صح: جرس نقي بنغمتين
    t = T(d); s = np.zeros(len(t))
    for f, st in ((1318.5, 0.0), (1975.5, 0.09)):
        tt = np.clip(t - st, 0, None); s += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.22) * (t >= st)
    return norm(s)

def g_register(d=0.8):                     # كاشير: خشخشة درج + جرس
    t = T(d); s = lp(rng.randn(len(t)), 0.4, 0.1) * np.exp(-t / 0.05) * 0.7
    b = np.clip(t - 0.08, 0, None)
    s += (np.sin(2 * np.pi * 2093 * b) + 0.5 * np.sin(2 * np.pi * 3136 * b)) * np.exp(-b / 0.25) * (t >= 0.08)
    return norm(s)

def g_coin(d=0.4):
    t = T(d); s = np.sin(2 * np.pi * 988 * t) * (t < 0.07) + np.sin(2 * np.pi * 1319 * t) * (t >= 0.07)
    return norm(s * np.exp(-t / 0.12))

def g_shutter(d=0.18):                     # غالق كاميرا: نقرتين قريبين
    s = np.zeros(int(d * SR)); k = g_click(0.04)
    for p, g in ((0.0, 1.0), (0.07, 0.8)):
        i = int(p * SR); s[i:i + len(k)] += k * g
    return norm(s)

def g_riser(d=1.2):                        # توتر بيطلع لفوق
    t = T(d); f = 180 * np.exp(np.log(900 / 180) * t / d)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.5 + lp(rng.randn(len(t)), 0.02, 0.35) * 0.5
    return norm(s * (t / d) ** 2)

def g_stamp(d=0.35):
    t = T(d); s = g_thud(160, 70, d) * 0.8
    s[:int(0.02 * SR)] += lp(rng.randn(int(0.02 * SR)), 0.8, 0.3) * 0.8
    return norm(s)

GEN = {
    "whoosh": lambda: g_whoosh(True), "whoosh_up": lambda: g_whoosh(True, 0.34),
    "whoosh_down": lambda: g_whoosh(False, 0.30),
    "thud": g_thud, "impact": g_thud, "hammer": lambda: g_thud(110, 45, 0.4), "stamp": g_stamp,
    "tap": g_tap, "click": g_click, "tick": g_tick, "keys": g_keys, "pop": g_pop, "pen": g_pen,
    "card": g_card, "thin": g_thin, "register": g_register, "coin": g_coin,
    "shutter": g_shutter, "riser": g_riser,
}


# ── تحميل ملف حقيقي إن وُجد ──────────────────────────────────────────────
def find_file(kind):
    for d in (S + "sfx-lib", os.path.join(SKILL, "assets", "sfx")):
        for e in ("wav", "mp3", "m4a", "aac", "aif", "aiff", "caf"):
            hit = glob.glob(os.path.join(d, f"{kind}.{e}"))
            if hit: return hit[0]
    return None

def load_file(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return norm(np.frombuffer(raw, dtype="<f4").astype(np.float64))

_cache = {}
def sound(kind):
    if kind not in _cache:
        f = find_file(kind)
        _cache[kind] = (load_file(f), os.path.basename(f)) if f else (GEN[kind](), "مولَّد")
    return _cache[kind]


# ── التجميع ──────────────────────────────────────────────────────────────
cfg = json.load(open(S + "sfx.json"))
levels = {**LEVELS, **cfg.get("levels", {})}
master = float(cfg.get("gain", 0))
kinds = [k for k, v in cfg.items() if isinstance(v, list) and k in GEN]
unknown = [k for k, v in cfg.items() if isinstance(v, list) and k not in GEN]

if LIST:
    for k in sorted(GEN):
        f = find_file(k)
        print(f"{k:12s} {levels.get(k, -18):>4} dB   {os.path.basename(f) if f else 'مولَّد'}")
    sys.exit(0)

caps = json.load(open(S + "caps.json"))
DUR = caps["total"] + cfg["outro"]
n = int(DUR * SR) + SR
buf = np.zeros(n)

def add(sig, t0, g):
    i = max(0, int(t0 * SR)); j = min(n, i + len(sig)); buf[i:j] += sig[:j - i] * g

count = {}
for k in kinds:
    sig, src = sound(k)
    g = 10 ** ((levels.get(k, -18) + master) / 20)
    for t0 in cfg[k]:
        add(sig, t0, g)
    count[k] = (len(cfg[k]), levels.get(k, -18) + master, src)

buf = np.clip(buf, -0.95, 0.95)
pcm = (buf * 32767).astype("<i2")
import wave
w = wave.open(S + "sfx.wav", "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes(np.repeat(pcm[:, None], 2, axis=1).ravel().tobytes()); w.close()

peak = float(np.max(np.abs(buf)))
hits = sum(len(cfg[k]) for k in kinds if k in HITS)
per_min = hits / max(caps["total"] / 60, 1e-9)
print(f"sfx ok · ذروة {20*np.log10(peak+1e-9):.1f} dBFS · مدة {len(pcm)/SR:.2f}s")
for k, (c, db, src) in sorted(count.items(), key=lambda x: -x[1][1]):
    print(f"  {k:12s} ×{c:<3} {db:>5.1f} dB  ({src})")
print(f"  الضربات: {hits} ← {per_min:.1f} بالدقيقة" + ("  ⚠️ أكتر من 15 — قلّلها" if per_min > 15 else ""))
if peak > 10 ** (-8 / 20): print("  ⚠️ الذروة فوق ‎-8 dBFS — فيه أصوات متراكبة، نزّل gain أو باعد بينها")
if unknown: print("  ⚠️ أنواع مش معروفة اتجاهلت:", ", ".join(unknown))
