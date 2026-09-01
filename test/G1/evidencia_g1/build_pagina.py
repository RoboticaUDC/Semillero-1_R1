#!/usr/bin/env python3
"""Construye evidencia_g1/pagina.html: resumen visual autocontenido (data URIs)."""
import os, glob, base64, io, json
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(ROOT, *p)

def b64_jpeg(img, ancho, q=82):
    im = Image.open(img) if isinstance(img, str) else img
    im = im.convert("RGB")
    im = im.resize((ancho, int(ancho * im.height / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def b64_gif(mp4, ancho=340, fps=8, segundos=4.5):
    fr = imageio.mimread(mp4, memtest=False)[:int(50 * segundos)]
    paso = max(1, int(round(50 / fps)))
    peq = [np.asarray(Image.fromarray(f).resize((ancho, int(ancho * f.shape[0] / f.shape[1])), Image.LANCZOS))
           for f in fr[::paso]]
    buf = io.BytesIO()
    imageio.mimsave(buf, peq, format="GIF", duration=1000/fps, loop=0)
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode()

corridas = []
for f in sorted(glob.glob(D("data", "*.npz"))):
    z = np.load(f, allow_pickle=True)
    n = os.path.basename(f)[:-4]
    tel = z["tel"]
    corridas.append(dict(
        nombre=n, titulo=str(z["titulo"]).split(" - ", 1)[1], archivo=str(z["archivo"]),
        desc=str(z["desc"]), caido=float(z["caido"]), dur=float(tel[-1, 0]),
        zmin=float(tel[:, 1].min()), taumax=float(tel[:, 6].max()),
        num=n[:2],
        clase=("banco" if float(z["caido"]) < 0 else ("inducido" if "fallo" in n else "nominal")),
        thumb=b64_jpeg(D("frames", n + "_final.png"), 520),
    ))

datos = dict(
    corridas=corridas,
    altura=b64_jpeg(D("graficas", "altura_pelvis.png"), 1180),
    tiempos=b64_jpeg(D("graficas", "tiempo_caida.png"), 900),
    comparativa=b64_gif(D("media", "comparativa_2x2.mp4"), 460, 8),
)
with open(D("_datos_pagina.json"), "w") as fh:
    json.dump(datos, fh)
mb = os.path.getsize(D("_datos_pagina.json")) / 1e6
print(f"assets listos: {len(corridas)} corridas, {mb:.1f} MB")
import json, os, html
ROOT = os.path.dirname(os.path.abspath(__file__)) or "."
d = json.load(open(os.path.join(ROOT, "_datos_pagina.json")))
C = d["corridas"]
ESC = html.escape

caidos = [c for c in C if c["caido"] >= 0]
t_min = min(c["caido"] for c in caidos)
t_max = max(c["caido"] for c in caidos)
ESCALA = 16.0

def barra(c):
    val = c["caido"] if c["caido"] >= 0 else c["dur"]
    pct = max(1.2, 100 * val / ESCALA)
    est = "caida" if c["caido"] >= 0 else "sincaida"
    txt = f"{c['caido']:.2f} s" if c["caido"] >= 0 else f"sin caída · {c['dur']:.0f} s"
    return f'''<div class="fila">
        <span class="fila-num">{c['num']}</span>
        <span class="fila-nom">{ESC(c['titulo'])}</span>
        <span class="fila-pista"><span class="fila-barra {est} {c['clase']}" style="width:{pct:.1f}%"></span></span>
        <span class="fila-val">{txt}</span>
      </div>'''

def tarjeta(c):
    if c["caido"] >= 0:
        chip = f'<span class="chip chip-bad">caída · {c["caido"]:.2f} s</span>'
    else:
        chip = '<span class="chip chip-ok">sin caída</span>'
    etiqueta = {"nominal": "código tal cual", "inducido": "fallo inducido",
                "banco": "banco de pruebas"}[c["clase"]]
    return f'''<figure class="tarjeta {c['clase']}">
        <img src="{c['thumb']}" alt="Fotograma final de {ESC(c['titulo'])}" loading="lazy">
        <figcaption>
          <div class="tj-cab"><span class="tj-num">{c['num']}</span>{chip}</div>
          <h3>{ESC(c['titulo'])}</h3>
          <p>{ESC(c['desc'])}</p>
          <div class="tj-pie"><code>{ESC(c['archivo'])}</code><span class="tj-tipo">{etiqueta}</span></div>
        </figcaption>
      </figure>'''

HALLAZGOS = [
    ("El PD no tiene lazo de equilibrio",
     "<code>play_amo_stable.py</code> persigue <code>BASE_POSE</code> articulación por articulación y "
     "nada realimenta el roll/pitch ni la velocidad de la base. El robot se inclina y cae siempre "
     "cerca de los 2.8 s. Subir el límite de par de 60 a 200 Nm no cambia el resultado: no es "
     "saturación, es ausencia de estrategia de tobillo y cadera."),
    ("La observación de play_amo.py no coincide con la que lee la red",
     "La traza de <code>amo_jit.pt</code> segmenta su entrada así: repite las últimas 372 columnas al "
     "frente, inserta 105 ceros en la columna 465, y luego lee <code>obs_prop = obs[0:93]</code>, "
     "<code>obs_demo = obs[198:215]</code>, <code>obs_priv = obs[215:218]</code> e historial "
     "<code>[-1, 10, 93]</code>. El script arma en cambio qpos(30)+qvel(29)+adapter(15)+acción(15)+fase(4) "
     "con historial de 11 y los 20 comandos al final: los comandos nunca caen en el tramo "
     "<code>obs_demo</code> que la red realmente usa."),
    ("La policy está atada a cuda:0 dentro del grafo",
     "<code>amo_jit.pt</code> se trazó con <code>device=cuda:0</code> escrito en el TorchScript, así que "
     "no corre en CPU. Además el entorno <code>r1deploy</code> (torch 2.3.1+cu121) no tiene kernels para "
     "la RTX 5060 Ti. La evidencia se generó con torch 2.7.0+cu128 en <code>env_isaaclab</code>, "
     "instalando MuJoCo ahí."),
    ("ArmController.py está indexado para el R1",
     "Usa 10 joints de brazo en <code>qpos[21:31]</code> y escribe en <code>ctrl[14:24]</code>. El G1 "
     "tiene 8 joints de brazo en <code>qpos[22:30]</code> / <code>ctrl[15:23]</code>, de modo que esa "
     "escritura pisa el <code>waist_pitch</code>. El arnés de evidencia reimplementa las poses con la "
     "indexación del G1."),
]

hall_html = "\n".join(
    f'<article class="hallazgo"><h3>{ESC(t)}</h3><p>{c}</p></article>' for t, c in HALLAZGOS)

INVENTARIO = [
    ("media/*.mp4", "14 escenarios · 960×540 · 50 fps · con HUD de telemetría"),
    ("media/*.gif", "los mismos escenarios · 560 px · 20 fps"),
    ("media/gif_ligero/*.gif", "versiones compactas · 440 px · 12 fps · 3–9 MB"),
    ("media/comparativa_2x2.*", "cuatro modos de fallo en paralelo"),
    ("frames/*.png", "fotograma de inicio, medio y final de cada corrida"),
    ("graficas/*.png", "altura de pelvis, supervivencia, inclinación, mosaico"),
    ("data/*.npz", "telemetría cruda: t, z, roll, pitch, x, y, ‖τ‖, ‖qvel‖"),
]
inv_html = "\n".join(
    f'<tr><td><code>{ESC(a)}</code></td><td>{ESC(b)}</td></tr>' for a, b in INVENTARIO)

PAGINA = f'''<title>Evidencia de simulación · Unitree G1</title>
<style>
:root {{
  --bg:#eceff3; --surface:#ffffff; --surface-2:#f5f7fa; --ink:#121821; --muted:#59637a;
  --line:#d3dae4; --line-fuerte:#b9c3d1; --accent:#9a5a12; --accent-suave:#f0e2cf;
  --ok:#166b4c; --bad:#a52a37; --sombra:0 1px 2px rgba(18,24,33,.06), 0 8px 24px rgba(18,24,33,.05);
  --sans:ui-sans-serif,"DejaVu Sans","Segoe UI",system-ui,-apple-system,sans-serif;
  --mono:ui-monospace,"DejaVu Sans Mono","SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#0c0f14; --surface:#151a22; --surface-2:#1b212b; --ink:#e3e9f3; --muted:#8d99ae;
    --line:#232b37; --line-fuerte:#333d4c; --accent:#d9a04b; --accent-suave:#2a2317;
    --ok:#48b385; --bad:#e2606a; --sombra:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0c0f14; --surface:#151a22; --surface-2:#1b212b; --ink:#e3e9f3; --muted:#8d99ae;
  --line:#232b37; --line-fuerte:#333d4c; --accent:#d9a04b; --accent-suave:#2a2317;
  --ok:#48b385; --bad:#e2606a; --sombra:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased;
}}
.envoltura {{ max-width:1080px; margin:0 auto; padding:0 24px 80px; }}
.eyebrow {{
  font-family:var(--mono); font-size:11.5px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--muted); margin:0;
}}
h1 {{ font-size:clamp(30px,4.4vw,46px); line-height:1.08; letter-spacing:-.022em; margin:14px 0 0;
     text-wrap:balance; font-weight:700; }}
h2 {{ font-size:24px; letter-spacing:-.012em; margin:0; font-weight:650; }}
h3 {{ font-size:16.5px; margin:0; letter-spacing:-.005em; font-weight:650; }}
p {{ margin:0; }}
code {{ font-family:var(--mono); font-size:.88em; background:var(--surface-2);
        border:1px solid var(--line); border-radius:4px; padding:1px 5px; }}

header.cab {{ padding:56px 0 30px; border-bottom:2px solid var(--ink); }}
.cab-sub {{ margin-top:16px; max-width:64ch; color:var(--muted); font-size:16.5px; }}
.metricas {{ display:flex; flex-wrap:wrap; gap:36px; margin-top:30px; }}
.metrica .v {{ font-family:var(--mono); font-size:29px; font-weight:700; letter-spacing:-.02em;
               font-variant-numeric:tabular-nums; display:block; }}
.metrica .k {{ font-family:var(--mono); font-size:11px; letter-spacing:.13em; text-transform:uppercase;
               color:var(--muted); }}
.metrica.v-bad .v {{ color:var(--bad); }}
.metrica.v-ok .v {{ color:var(--ok); }}

section {{ margin-top:56px; }}
.sec-cab {{ display:flex; align-items:baseline; gap:14px; padding-bottom:12px;
            border-bottom:1px solid var(--line-fuerte); margin-bottom:26px; }}
.sec-cab .eyebrow {{ margin-left:auto; }}

.tabla-barras {{ display:flex; flex-direction:column; gap:7px; }}
.fila {{ display:grid; grid-template-columns:30px minmax(120px,1.25fr) 3fr 92px; gap:14px;
         align-items:center; font-size:14px; }}
.fila-num {{ font-family:var(--mono); color:var(--muted); font-size:12.5px; }}
.fila-nom {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.fila-pista {{ background:var(--surface-2); border:1px solid var(--line); border-radius:2px; height:16px;
               position:relative; overflow:hidden; }}
.fila-barra {{ display:block; height:100%; background:var(--bad); opacity:.85; }}
.fila-barra.nominal {{ background:var(--accent); }}
.fila-barra.sincaida {{ background:var(--ok); }}
.fila-val {{ font-family:var(--mono); font-size:12.5px; text-align:right;
             font-variant-numeric:tabular-nums; color:var(--muted); }}
.leyenda {{ display:flex; gap:20px; flex-wrap:wrap; margin-top:18px; font-family:var(--mono);
            font-size:11.5px; letter-spacing:.06em; color:var(--muted); text-transform:uppercase; }}
.punto {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:7px; }}

.rejilla {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:20px; }}
.tarjeta {{ margin:0; background:var(--surface); border:1px solid var(--line); border-radius:8px;
            overflow:hidden; box-shadow:var(--sombra); display:flex; flex-direction:column; }}
.tarjeta img {{ width:100%; display:block; aspect-ratio:16/9; object-fit:cover;
                border-bottom:1px solid var(--line); }}
.tarjeta figcaption {{ padding:15px 16px 14px; display:flex; flex-direction:column; gap:9px; flex:1; }}
.tarjeta.inducido {{ border-top:3px solid var(--bad); }}
.tarjeta.nominal {{ border-top:3px solid var(--accent); }}
.tarjeta.banco {{ border-top:3px solid var(--ok); }}
.tj-cab {{ display:flex; align-items:center; gap:10px; }}
.tj-num {{ font-family:var(--mono); font-size:12.5px; color:var(--muted); }}
.chip {{ font-family:var(--mono); font-size:11px; letter-spacing:.05em; padding:2.5px 8px;
         border-radius:100px; margin-left:auto; font-variant-numeric:tabular-nums; }}
.chip-bad {{ color:var(--bad); border:1px solid var(--bad); }}
.chip-ok {{ color:var(--ok); border:1px solid var(--ok); }}
.tarjeta p {{ color:var(--muted); font-size:13.6px; line-height:1.5; }}
.tj-pie {{ margin-top:auto; padding-top:11px; border-top:1px solid var(--line); display:flex;
           gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap; }}
.tj-pie code {{ background:none; border:none; padding:0; color:var(--muted); font-size:11.5px; }}
.tj-tipo {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
            color:var(--accent); }}

.hallazgos {{ display:flex; flex-direction:column; gap:22px; }}
.hallazgo {{ border-left:2px solid var(--accent); padding:2px 0 2px 20px; }}
.hallazgo p {{ color:var(--muted); font-size:15px; margin-top:7px; max-width:72ch; }}

figure.grafica {{ margin:0; background:var(--surface); border:1px solid var(--line); border-radius:8px;
                  padding:14px; box-shadow:var(--sombra); }}
figure.grafica img {{ width:100%; display:block; border-radius:4px; }}
figure.grafica figcaption {{ font-family:var(--mono); font-size:11.5px; color:var(--muted);
                             letter-spacing:.05em; margin-top:11px; text-transform:uppercase; }}
.par {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}

table.inv {{ width:100%; border-collapse:collapse; font-size:14.5px; }}
table.inv td {{ padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
table.inv tr td:first-child {{ width:230px; white-space:nowrap; }}
table.inv td code {{ background:none; border:none; padding:0; color:var(--accent); }}
.envolt-tabla {{ overflow-x:auto; }}

pre.cmd {{ background:var(--surface); border:1px solid var(--line); border-left:2px solid var(--accent);
           border-radius:6px; padding:15px 18px; overflow-x:auto; font-family:var(--mono);
           font-size:13px; line-height:1.85; margin:0; color:var(--ink); }}
pre.cmd .c {{ color:var(--muted); }}
footer {{ margin-top:60px; padding-top:22px; border-top:1px solid var(--line); color:var(--muted);
          font-family:var(--mono); font-size:11.5px; letter-spacing:.06em; }}
@media (max-width:720px) {{
  .par {{ grid-template-columns:1fr; }}
  .fila {{ grid-template-columns:26px 1.6fr 68px; }}
  .fila-pista {{ display:none; }}
}}
</style>

<div class="envoltura">
  <header class="cab">
    <p class="eyebrow">Registro de simulación · MuJoCo · Unitree G1 23-DoF</p>
    <h1>Evidencia de los controladores del G1: 14 corridas, 13 caídas</h1>
    <p class="cab-sub">Material generado sin ventana (render offscreen) a partir del código del
      repositorio AMO. Cinco corridas ejecutan el código tal como está hoy; ocho degradan un
      parámetro a propósito para aislar cada modo de fallo; una fija la pelvis para validar las poses
      de brazos. Nada de esto corresponde al R1.</p>
    <div class="metricas">
      <div class="metrica"><span class="v">14</span><span class="k">corridas</span></div>
      <div class="metrica v-bad"><span class="v">13</span><span class="k">terminan en caída</span></div>
      <div class="metrica v-bad"><span class="v">{t_min:.2f} s</span><span class="k">caída más rápida</span></div>
      <div class="metrica"><span class="v">{t_max:.2f} s</span><span class="k">caída más lenta</span></div>
      <div class="metrica v-ok"><span class="v">16 s</span><span class="k">banco de brazos sin caída</span></div>
    </div>
  </header>

  <section>
    <div class="sec-cab"><h2>Tiempo de supervivencia</h2><p class="eyebrow">segundos hasta z &lt; 0.45 m</p></div>
    <div class="tabla-barras">
      {"".join(barra(c) for c in sorted(C, key=lambda c: c["caido"] if c["caido"] >= 0 else 1e9))}
    </div>
    <div class="leyenda">
      <span><span class="punto" style="background:var(--accent)"></span>código tal cual</span>
      <span><span class="punto" style="background:var(--bad)"></span>fallo inducido</span>
      <span><span class="punto" style="background:var(--ok)"></span>sin caída</span>
    </div>
  </section>

  <section>
    <div class="sec-cab"><h2>Cuatro fallos en paralelo</h2><p class="eyebrow">media/comparativa_2x2.mp4</p></div>
    <figure class="grafica">
      <img src="{d['comparativa']}" alt="Comparativa 2x2 de cuatro modos de fallo del G1">
      <figcaption>Arriba: policy AMO tal cual · ruido de 22 Nm en los pares. Abajo: 45 ms de latencia · pose inicial sin flexión.</figcaption>
    </figure>
  </section>

  <section>
    <div class="sec-cab"><h2>Las 14 corridas</h2><p class="eyebrow">fotograma final de cada una</p></div>
    <div class="rejilla">
      {"".join(tarjeta(c) for c in C)}
    </div>
  </section>

  <section>
    <div class="sec-cab"><h2>Telemetría</h2><p class="eyebrow">graficas/</p></div>
    <div class="par">
      <figure class="grafica">
        <img src="{d['altura']}" alt="Altura de la pelvis frente al tiempo para los 14 escenarios">
        <figcaption>Altura de la pelvis · la x marca el instante de caída</figcaption>
      </figure>
      <figure class="grafica">
        <img src="{d['tiempos']}" alt="Barras de tiempo de supervivencia por escenario">
        <figcaption>Segundos hasta la caída, ordenados</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <div class="sec-cab"><h2>Qué explica los fallos</h2><p class="eyebrow">diagnóstico</p></div>
    <div class="hallazgos">
      {hall_html}
    </div>
  </section>

  <section>
    <div class="sec-cab"><h2>Archivos generados</h2><p class="eyebrow">evidencia_g1/</p></div>
    <div class="envolt-tabla"><table class="inv"><tbody>
      {inv_html}
    </tbody></table></div>
  </section>

  <section>
    <div class="sec-cab"><h2>Reproducir</h2><p class="eyebrow">env_isaaclab · torch 2.7.0+cu128</p></div>
    <pre class="cmd"><span class="c"># todos los escenarios</span>
python evidencia_g1/render_g1.py

<span class="c"># solo algunos, por prefijo</span>
python evidencia_g1/render_g1.py 05 09 13

<span class="c"># gráficas, mosaico y comparativa</span>
python evidencia_g1/resumen_g1.py</pre>
  </section>

  <footer>Generado el 2026-08-18 · render_g1.py + resumen_g1.py · g1.xml, amo_jit.pt, adapter_jit.pt</footer>
</div>
'''

with open(os.path.join(ROOT, "pagina.html"), "w") as fh:
    fh.write(PAGINA)
print("pagina.html  %.1f MB" % (os.path.getsize(os.path.join(ROOT, "pagina.html")) / 1e6))
