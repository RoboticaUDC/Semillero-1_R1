#!/usr/bin/env python3
"""
resumen_g1.py — Graficas y montajes a partir de la telemetria de render_g1.py.

Genera:
  graficas/altura_pelvis.png      altura de la pelvis vs tiempo (todos)
  graficas/tiempo_caida.png       barras: segundos hasta la caida
  graficas/inclinacion.png        roll/pitch de la base vs tiempo
  graficas/mosaico_fallos.png     contact sheet con el instante final de cada corrida
  media/comparativa_2x2.mp4/.gif  cuatro modos de fallo en paralelo
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, MEDIA, FRAMES, GRAF = (os.path.join(ROOT, d) for d in ("data", "media", "frames", "graficas"))
os.makedirs(GRAF, exist_ok=True)

FDIR = "/usr/share/fonts/truetype/dejavu"
F_T = ImageFont.truetype(f"{FDIR}/DejaVuSans-Bold.ttf", 20)
F_S = ImageFont.truetype(f"{FDIR}/DejaVuSans.ttf", 15)

plt.rcParams.update({
    "figure.facecolor": "#0f1115", "axes.facecolor": "#161a21",
    "axes.edgecolor": "#3a4150", "grid.color": "#252b36",
    "text.color": "#dde4f0", "axes.labelcolor": "#dde4f0",
    "xtick.color": "#9aa7bd", "ytick.color": "#9aa7bd",
    "font.size": 10, "axes.titlesize": 12,
})


def cargar():
    regs = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        regs.append(dict(
            nombre=os.path.basename(f)[:-4], tel=z["tel"],
            titulo=str(z["titulo"]), desc=str(z["desc"]),
            archivo=str(z["archivo"]), esperado=str(z["esperado"]),
            caido=float(z["caido"])))
    return regs


def color_de(r):
    return "#e05252" if "fallo" in r["nombre"] else ("#4fa3ff" if r["caido"] >= 0 else "#3fb96b")


def g_altura(regs):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for r in regs:
        t, z = r["tel"][:, 0], r["tel"][:, 1]
        ax.plot(t, z, lw=1.7, color=color_de(r), alpha=0.9,
                label=f"{r['nombre'][:2]} {r['titulo'].split(' - ')[1][:34]}")
        if r["caido"] >= 0:
            ax.plot(r["caido"], np.interp(r["caido"], t, z), "x", color="#ff6b6b", ms=8, mew=2)
    ax.axhline(0.45, color="#ffb03a", ls="--", lw=1.2)
    ax.text(0.06, 0.465, "umbral de caida (0.45 m)", color="#ffb03a", fontsize=9)
    ax.axhline(0.793, color="#5a6478", ls=":", lw=1.2)
    ax.text(0.06, 0.805, "altura nominal de pie (0.793 m)", color="#8b97ad", fontsize=9)
    ax.set_xlabel("tiempo (s)"); ax.set_ylabel("altura de la pelvis  z (m)")
    ax.set_title("G1 - altura de la pelvis por escenario  (x = instante de caida)")
    ax.grid(alpha=0.35); ax.set_ylim(-0.05, 1.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(GRAF, "altura_pelvis.png"), dpi=140)
    plt.close(fig)


def g_tiempo(regs):
    rs = sorted(regs, key=lambda r: (r["caido"] if r["caido"] >= 0 else 1e9))
    nom = [r["nombre"].replace("_", " ") for r in rs]
    val = [r["caido"] if r["caido"] >= 0 else r["tel"][-1, 0] for r in rs]
    col = ["#e05252" if r["caido"] >= 0 else "#3fb96b" for r in rs]
    fig, ax = plt.subplots(figsize=(10, 6))
    b = ax.barh(nom, val, color=col, height=0.62)
    for rect, r, v in zip(b, rs, val):
        txt = f"{v:.2f} s" if r["caido"] >= 0 else f"sin caida ({v:.0f} s)"
        ax.text(v + 0.08, rect.get_y() + rect.get_height() / 2, txt,
                va="center", fontsize=9, color="#dde4f0")
    ax.set_xlabel("segundos hasta la caida"); ax.invert_yaxis()
    ax.set_title("G1 - tiempo de supervivencia por escenario")
    ax.grid(axis="x", alpha=0.3); ax.set_xlim(0, max(val) * 1.28)
    fig.tight_layout(); fig.savefig(os.path.join(GRAF, "tiempo_caida.png"), dpi=140)
    plt.close(fig)


def g_inclinacion(regs):
    n = len(regs)
    filas = (n + 2) // 3
    fig, axes = plt.subplots(filas, 3, figsize=(13, 2.5 * filas), squeeze=False)
    for ax, r in zip(axes.ravel(), regs):
        t = r["tel"][:, 0]
        ax.plot(t, np.degrees(r["tel"][:, 2]), lw=1.3, color="#4fa3ff", label="roll")
        ax.plot(t, np.degrees(r["tel"][:, 3]), lw=1.3, color="#ffb03a", label="pitch")
        if r["caido"] >= 0:
            ax.axvline(r["caido"], color="#e05252", ls="--", lw=1.1)
        ax.set_title(r["nombre"][:26], fontsize=9)
        ax.grid(alpha=0.3); ax.set_ylim(-190, 190)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("G1 - inclinacion de la base (grados)", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(GRAF, "inclinacion.png"), dpi=140)
    plt.close(fig)


def mosaico(regs, cw=420):
    ch = int(cw * 540 / 960)
    cols = 4
    filas = (len(regs) + cols - 1) // cols
    alto_txt = 46
    W, H = cols * cw, filas * (ch + alto_txt)
    lienzo = Image.new("RGB", (W, H), (12, 14, 20))
    dr = ImageDraw.Draw(lienzo)
    for i, r in enumerate(regs):
        f = os.path.join(FRAMES, r["nombre"] + "_final.png")
        if not os.path.exists(f):
            continue
        x, y = (i % cols) * cw, (i // cols) * (ch + alto_txt)
        lienzo.paste(Image.open(f).resize((cw, ch), Image.LANCZOS), (x, y))
        est = f"CAIDA t={r['caido']:.2f} s" if r["caido"] >= 0 else "SIN CAIDA"
        col = (224, 82, 82) if r["caido"] >= 0 else (63, 185, 107)
        dr.text((x + 10, y + ch + 6), r["nombre"].replace("_", " "), font=F_T, fill=(226, 233, 245))
        dr.text((x + 10, y + ch + 28), f"{r['archivo']}  |  {est}", font=F_S, fill=col)
    lienzo.save(os.path.join(GRAF, "mosaico_fallos.png"))


def comparativa(nombres, salida="comparativa_2x2"):
    vids = []
    for n in nombres:
        p = os.path.join(MEDIA, n + ".mp4")
        if os.path.exists(p):
            vids.append((n, imageio.mimread(p, memtest=False)))
    if len(vids) < 4:
        return
    largo = max(len(v) for _, v in vids)
    cw, ch = 480, 270
    out = []
    for i in range(largo):
        lienzo = Image.new("RGB", (cw * 2, ch * 2), (10, 12, 16))
        for k, (n, fr) in enumerate(vids[:4]):
            img = fr[min(i, len(fr) - 1)]
            lienzo.paste(Image.fromarray(img).resize((cw, ch), Image.LANCZOS),
                         ((k % 2) * cw, (k // 2) * ch))
        out.append(np.asarray(lienzo))
    imageio.mimsave(os.path.join(MEDIA, salida + ".mp4"), out, fps=50,
                    quality=7, macro_block_size=None)
    peq = [np.asarray(Image.fromarray(f).resize((720, 405), Image.LANCZOS)) for f in out[::3]]
    imageio.mimsave(os.path.join(MEDIA, salida + ".gif"), peq, duration=1000 / 16, loop=0)


if __name__ == "__main__":
    regs = cargar()
    print(f"[resumen_g1] {len(regs)} corridas")
    g_altura(regs); print("  graficas/altura_pelvis.png")
    g_tiempo(regs); print("  graficas/tiempo_caida.png")
    g_inclinacion(regs); print("  graficas/inclinacion.png")
    mosaico(regs); print("  graficas/mosaico_fallos.png")
    comparativa(["05_g1_policy_amo", "07_fallo_ruido_pares",
                 "09_fallo_latencia", "10_fallo_pose_inicial"])
    print("  media/comparativa_2x2.mp4 / .gif")


def gifs_ligeros(ancho=440, fps=12):
    """Versiones compactas para pegar en documentos o chat."""
    dst = os.path.join(MEDIA, "gif_ligero")
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(MEDIA, "*.mp4"))):
        n = os.path.basename(p)[:-4]
        fr = imageio.mimread(p, memtest=False)
        paso = max(1, int(round(50 / fps)))
        peq = [np.asarray(Image.fromarray(f).resize((ancho, int(ancho * f.shape[0] / f.shape[1])),
                                                    Image.LANCZOS)) for f in fr[::paso]]
        sal = os.path.join(dst, n + ".gif")
        imageio.mimsave(sal, peq, duration=1000 / fps, loop=0)
        print(f"  gif_ligero/{n}.gif  {os.path.getsize(sal)/1e6:.1f} MB")
