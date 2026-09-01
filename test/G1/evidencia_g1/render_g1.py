#!/usr/bin/env python3
"""
render_g1.py — Generador de evidencia visual (MP4 / GIF / PNG) del robot G1.

Renderiza escenarios en MuJoCo sin ventana (offscreen) y produce:
  - evidencia_g1/media/<escenario>.mp4
  - evidencia_g1/media/<escenario>.gif
  - evidencia_g1/frames/<escenario>_t*.png
  - evidencia_g1/data/<escenario>.npz   (telemetria para graficas)

Escenarios: los que funcionan (PD estable, poses de brazos, policy AMO) y
los degradados a proposito (ruido, ganancias bajas, latencia, pose inicial
mala, adapter desconectado) para documentar el proceso completo.

Uso:
    python render_g1.py                 # todos los escenarios
    python render_g1.py 01 06 09        # solo los que empiecen por ese prefijo
    python render_g1.py --lista
"""

import os
import sys
import math
import re
import shutil

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
OUT_MEDIA = os.path.join(ROOT, "media")
OUT_FRAMES = os.path.join(ROOT, "frames")
OUT_DATA = os.path.join(ROOT, "data")
for d in (OUT_MEDIA, OUT_FRAMES, OUT_DATA):
    os.makedirs(d, exist_ok=True)

W, H = 960, 540
FPS = 50               # frames de video por segundo simulado
RENDER_EVERY = 20      # 1 frame cada 20 pasos (dt=0.001) -> 50 fps

# ---------------------------------------------------------------------------
# Pose base y ganancias (identicas a play_amo.py / play_amo_stable.py)
# ---------------------------------------------------------------------------
BASE_POSE = np.array([
    -0.35, 0.12, 0.00, 0.50, -0.25, -0.06,      # pierna izq
    -0.35, -0.12, 0.00, 0.50, -0.25, 0.06,      # pierna der
     0.00, 0.00, 0.00,                          # cintura
     0.30, 0.30, 0.00, 0.50,                    # brazo izq
     0.30, -0.30, 0.00, 0.50,                   # brazo der
], dtype=np.float32)

KP = np.array([200,150,150,300,100,60, 200,150,150,300,100,60,
               150,100,100, 80,60,60,80, 80,60,60,80], dtype=np.float32)
KD = np.array([5,4,4,6,3,2, 5,4,4,6,3,2,
               4,3,3, 2,2,2,2, 2,2,2,2], dtype=np.float32)

# Poses de brazos G1 (8 joints: L pitch/roll/yaw/elbow, R pitch/roll/yaw/elbow)
ARM_POSES = {
    "neutral":  np.array([ 0.30, 0.30, 0.00, 0.50,  0.30,-0.30, 0.00, 0.50], np.float32),
    "saludo_a": np.array([ 0.30, 0.30, 0.00, 0.50,  0.35,-1.56,-1.26,-0.40], np.float32),
    "saludo_b": np.array([ 0.30, 0.30, 0.00, 0.50,  0.35,-1.56,-1.26, 1.07], np.float32),
    "cruz":     np.array([ 0.00, 1.45, 0.00, 0.10,  0.00,-1.45, 0.00, 0.10], np.float32),
    "carga":    np.array([-0.10, 0.20, 0.00, 0.30, -0.10,-0.20, 0.00, 0.30], np.float32),
    "guardia":  np.array([ 0.60, 0.50, 0.00, 1.20,  0.60,-0.50, 0.00, 1.20], np.float32),
    "apunta":   np.array([ 0.30, 0.30, 0.00, 0.50, -0.10,-0.20, 0.00, 0.05], np.float32),
}

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def _font(name, size):
    path = os.path.join(FONT_DIR, name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


F_TITLE = _font("DejaVuSans-Bold.ttf", 26)
F_SUB = _font("DejaVuSans.ttf", 16)
F_MONO = _font("DejaVuSansMono-Bold.ttf", 17)
F_BADGE = _font("DejaVuSans-Bold.ttf", 20)
F_SMALL = _font("DejaVuSans.ttf", 13)


# ---------------------------------------------------------------------------
# Modelo con framebuffer offscreen ampliado
# ---------------------------------------------------------------------------
def load_model():
    src = os.path.join(PROJ, "g1.xml")
    with open(src, "r") as fh:
        xml = fh.read()
    if "offwidth" not in xml:
        xml = xml.replace("<global azimuth", f'<global offwidth="{W}" offheight="{H}" azimuth')
    if "skybox" not in xml:
        cielo = ('<asset><texture type="skybox" builtin="gradient" '
                 'rgb1="0.24 0.33 0.47" rgb2="0.05 0.06 0.09" width="256" height="1536"/></asset>')
        xml = xml.replace("<visual>", cielo + "\n  <visual>", 1)
    tmp = os.path.join(PROJ, ".g1_render_tmp.xml")
    with open(tmp, "w") as fh:
        fh.write(xml)
    model = mujoco.MjModel.from_xml_path(tmp)
    os.remove(tmp)
    return model


# ---------------------------------------------------------------------------
# Controladores
# ---------------------------------------------------------------------------
def target_pose_pd(cmd, phase, arm_target):
    """Pose objetivo PD: altura de torso + patron de marcha (play_amo_stable.py)."""
    t = BASE_POSE.copy()

    h = cmd.get("height", 0.0)
    t[0] += h * 0.6
    t[6] += h * 0.6
    t[3] -= h * 1.2
    t[9] -= h * 1.2
    t[4] -= h * 0.6
    t[10] -= h * 0.6

    vx, vy, vyaw = cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("vyaw", 0.0)
    if abs(vx) > 0.01 or abs(vy) > 0.01:
        amp = min(abs(vx) + abs(vy), 0.5) * 0.12 / 0.5
        sl, sr = math.sin(phase), math.sin(phase + math.pi)
        t[0] += sl * amp
        t[6] += sr * amp
        t[3] += max(0.0, -sl) * amp * 0.8
        t[9] += max(0.0, -sr) * amp * 0.8
        t[4] -= sl * amp * 0.4
        t[10] -= sr * amp * 0.4
        t[2] += vyaw * 0.1
        t[8] -= vyaw * 0.1
        t[1] += vy * 0.15
        t[7] += vy * 0.15

    if arm_target is not None:
        t[15:23] = arm_target
    return t


class Runner:
    """Ejecuta un escenario y devuelve frames + telemetria."""

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.data = mujoco.MjData(model)
        self.dt = model.opt.timestep
        self.f = cfg.get("fallos", {})

        self.cmd_pos = BASE_POSE.copy()
        self.phase = 0.0
        self.walk_phase = 0.0
        self.arm_pose = ARM_POSES["neutral"].copy()

        # buffer de latencia (estado retrasado)
        lat = int(self.f.get("latencia_ms", 0) / (self.dt * 1000))
        self.lat_n = lat
        self.state_buf = []

        self.rng = np.random.default_rng(cfg.get("semilla", 0))

        if cfg["controlador"] == "amo":
            self._init_policy()

        self.reset()

    # -- policy AMO ---------------------------------------------------------
    def _init_policy(self):
        import torch
        from collections import deque
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.policy = torch.jit.load(os.path.join(PROJ, "amo_jit.pt"), map_location=self.device).eval()
        self.adapter = torch.jit.load(os.path.join(PROJ, "adapter_jit.pt"), map_location=self.device).eval()
        st = torch.load(os.path.join(PROJ, "adapter_norm_stats.pt"), weights_only=False)
        self.in_mean = torch.tensor(st["input_mean"], device=self.device, dtype=torch.float32)
        self.in_std = torch.tensor(st["input_std"], device=self.device, dtype=torch.float32)
        self.out_mean = torch.tensor(st["output_mean"], device=self.device, dtype=torch.float32)
        self.out_std = torch.tensor(st["output_std"], device=self.device, dtype=torch.float32)
        self.obs_hist = deque(maxlen=11)
        self.extra_hist = deque(maxlen=25)
        self.adapter_out = np.zeros(15, np.float32)
        self.last_action = np.zeros(15, np.float32)
        self.commands20 = np.zeros(20, np.float32)

    def reset(self):
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, 0.793]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        mal = self.f.get("pose_inicial")
        if mal == "cero":
            d.qpos[7:30] = 0.0                      # piernas rectas, sin flexion
        elif mal == "inclinado":
            d.qpos[7:30] = BASE_POSE
            ang = 0.28
            d.qpos[3] = math.cos(ang / 2)
            d.qpos[5] = math.sin(ang / 2)           # pitch inicial hacia adelante
        elif mal == "alto":
            d.qpos[7:30] = BASE_POSE
            d.qpos[2] = 1.15                        # soltado desde el aire
        else:
            d.qpos[7:30] = BASE_POSE

        self.cmd_pos = d.qpos[7:30].astype(np.float32).copy()
        mujoco.mj_forward(m, d)

        if self.cfg["controlador"] == "amo":
            for _ in range(11):
                self.obs_hist.append(np.zeros(93, np.float32))
            for _ in range(25):
                self.extra_hist.append(np.zeros(93, np.float32))
            self.last_action[:] = 0.0
            self.adapter_out[:] = 0.0
            self.phase = 0.0

    # -- comandos por tiempo ------------------------------------------------
    def cmd_at(self, t):
        cmd = {}
        for t0, c in self.cfg.get("comandos", [(0.0, {})]):
            if t >= t0:
                cmd = c
        return cmd

    def arm_at(self, t):
        seq = self.cfg.get("brazos")
        if not seq:
            return None
        name = seq[0][1]
        for t0, n in seq:
            if t >= t0:
                name = n
        objetivo = ARM_POSES[name]
        self.arm_pose += 0.02 * (objetivo - self.arm_pose)   # interpolacion suave
        return self.arm_pose, name

    # -- lectura de estado (con latencia opcional) --------------------------
    def read_state(self):
        q = self.data.qpos[7:30].astype(np.float32)
        qd = self.data.qvel[6:29].astype(np.float32)
        if self.lat_n <= 0:
            return q, qd
        self.state_buf.append((q.copy(), qd.copy()))
        if len(self.state_buf) > self.lat_n:
            return self.state_buf.pop(0)
        return self.state_buf[0]

    # -- un paso ------------------------------------------------------------
    def step(self):
        t = self.data.time
        cmd = self.cmd_at(t)
        arm = self.arm_at(t)
        arm_target = arm[0] if arm else None

        if self.cfg["controlador"] == "pd":
            if abs(cmd.get("vx", 0)) > 0.01 or abs(cmd.get("vy", 0)) > 0.01:
                self.walk_phase += 2 * math.pi * 1.2 * self.dt
            objetivo = target_pose_pd(cmd, self.walk_phase, arm_target)
            self.cmd_pos += 0.08 * (objetivo - self.cmd_pos)
            q, qd = self.read_state()
            kp = KP * self.f.get("kp_escala", 1.0)
            kd = KD * self.f.get("kd_escala", 1.0)
            tau = kp * (self.cmd_pos - q) - kd * qd
            tau = np.clip(tau, -60, 60)
        else:
            tau = self.step_amo(cmd, arm_target)

        ruido = self.f.get("ruido_par", 0.0)
        if ruido > 0:
            tau = tau + self.rng.normal(0.0, ruido, size=tau.shape).astype(np.float32)

        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)
        if self.cfg.get("pelvis_fija"):
            self.data.qpos[0:3] = [0.0, 0.0, 0.95]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[0:6] = 0.0
            mujoco.mj_forward(self.model, self.data)
        return tau

    def step_amo(self, cmd, arm_target):
        torch = self.torch
        d = self.data
        qpos_t = torch.tensor(d.qpos, device=self.device, dtype=torch.float32)
        qvel_t = torch.tensor(d.qvel, device=self.device, dtype=torch.float32)

        if self.f.get("adapter_off"):
            self.adapter_out[:] = 0.0
        else:
            ain = torch.cat([qpos_t[:6], qvel_t[:6]]).unsqueeze(0)
            ain = (ain - self.in_mean) / (self.in_std + 1e-8)
            with torch.no_grad():
                aout = self.adapter(ain) * self.out_std + self.out_mean
            self.adapter_out = aout.squeeze().cpu().numpy()

        ph = np.array([math.sin(self.phase), math.cos(self.phase),
                       math.sin(self.phase + math.pi), math.cos(self.phase + math.pi)], np.float32)
        frame = np.concatenate([d.qpos.astype(np.float32), d.qvel.astype(np.float32),
                                self.adapter_out, self.last_action, ph])
        self.obs_hist.append(frame)
        self.extra_hist.append(frame)

        self.commands20[:] = 0.0
        self.commands20[0] = cmd.get("vx", 0.0)
        self.commands20[1] = cmd.get("vy", 0.0)
        self.commands20[2] = cmd.get("vyaw", 0.0)
        self.commands20[3] = cmd.get("height", 0.0)
        self.commands20[4] = cmd.get("pitch", 0.0)
        self.commands20[5] = cmd.get("roll", 0.0)
        self.commands20[6] = cmd.get("yaw", 0.0)

        obs = np.concatenate([np.array(self.obs_hist, np.float32).ravel(), self.commands20])
        extra = np.array(self.extra_hist, np.float32).ravel()
        with torch.no_grad():
            act = self.policy(torch.tensor(obs, device=self.device).unsqueeze(0),
                              torch.tensor(extra, device=self.device).unsqueeze(0)).squeeze()
        act = act.clamp(-40, 40).cpu().numpy() * self.f.get("escala_accion", 1.0)
        self.last_action = act.copy()

        # brazos con PD hacia BASE_POSE / pose comandada (igual que play_amo.py)
        obj_arm = arm_target if arm_target is not None else BASE_POSE[15:]
        q_arms = d.qpos[22:30].astype(np.float32)
        qd_arms = d.qvel[21:29].astype(np.float32)
        tau_arms = np.clip(80.0 * (obj_arm - q_arms) - 3.0 * qd_arms, -20, 20)

        tau = np.zeros(23, np.float32)
        tau[:15] = act
        tau[15:] = tau_arms
        self.phase += 2 * math.pi * 1.5 * self.dt
        if self.phase > 2 * math.pi:
            self.phase -= 2 * math.pi
        return tau


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def dibujar_hud(img, cfg, t, z, rp, cmd, estado, tau_n, arm_name, progreso):
    im = Image.fromarray(img).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")

    dr.rectangle([0, 0, W, 76], fill=(12, 14, 20, 200))
    dr.text((18, 12), cfg["titulo"], font=F_TITLE, fill=(240, 244, 252))
    dr.text((20, 46), cfg["desc"], font=F_SUB, fill=(150, 165, 190))

    ok = estado == "ESTABLE"
    col = (46, 160, 90) if ok else (198, 52, 52)
    etiqueta = "ESTABLE" if ok else "CAIDA"
    bw = 132
    dr.rounded_rectangle([W - bw - 18, 16, W - 18, 56], radius=8, fill=col + (235,))
    tw = dr.textlength(etiqueta, font=F_BADGE)
    dr.text((W - bw - 18 + (bw - tw) / 2, 25), etiqueta, font=F_BADGE, fill=(255, 255, 255))

    lines = [
        f"t      {t:5.2f} s",
        f"z base {z:5.3f} m",
        f"roll   {math.degrees(rp[0]):+6.1f} deg",
        f"pitch  {math.degrees(rp[1]):+6.1f} deg",
        f"vx cmd {cmd.get('vx', 0.0):+5.2f}",
        f"h  cmd {cmd.get('height', 0.0):+5.2f}",
        f"|tau|  {tau_n:6.1f} Nm",
    ]
    if arm_name:
        lines.append(f"brazos {arm_name}")
    bh = 22 * len(lines) + 20
    dr.rectangle([0, H - bh, 236, H], fill=(12, 14, 20, 190))
    for i, ln in enumerate(lines):
        dr.text((16, H - bh + 10 + 22 * i), ln, font=F_MONO, fill=(214, 226, 244))

    dr.rectangle([0, H - 5, W, H], fill=(30, 34, 44, 220))
    dr.rectangle([0, H - 5, int(W * progreso), H], fill=col + (255,))
    dr.text((W - 250, H - 26), cfg["archivo"] + "  |  MuJoCo G1 23-DoF",
            font=F_SMALL, fill=(150, 165, 190))
    return np.asarray(im)


def quat_a_euler(q):
    qw, qx, qy, qz = q
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    s = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, s) if abs(s) >= 1 else math.asin(s)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Ejecucion de un escenario
# ---------------------------------------------------------------------------
def correr(cfg, model, renderer):
    run = Runner(cfg, model)
    dur = cfg["duracion"]
    nsteps = int(dur / run.dt)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance = cfg.get("cam_dist", 3.2)
    cam.elevation = -14
    cam.azimuth = cfg.get("cam_az", 132)

    frames, tel = [], []
    caido_desde = None

    for i in range(nsteps):
        tau = run.step()
        if i % RENDER_EVERY:
            continue
        d = run.data
        z = float(d.qpos[2])
        rp = quat_a_euler(d.qpos[3:7])
        t = float(d.time)
        cmd = run.cmd_at(t)
        caido = z < 0.45 or abs(rp[0]) > 1.0 or abs(rp[1]) > 1.0
        if caido and caido_desde is None:
            caido_desde = t
        estado = "CAIDA" if caido_desde is not None else "ESTABLE"
        arm_name = run.arm_at(t)[1] if cfg.get("brazos") else None

        if cfg.get("cam_giro"):
            cam.azimuth = cfg.get("cam_az", 132) + 18 * math.sin(2 * math.pi * t / dur)
        renderer.update_scene(d, camera=cam)
        img = renderer.render()
        frames.append(dibujar_hud(img, cfg, t, z, rp, cmd, estado,
                                  float(np.linalg.norm(tau)), arm_name, t / dur))
        tel.append([t, z, rp[0], rp[1], float(d.qpos[0]), float(d.qpos[1]),
                    float(np.linalg.norm(tau)), float(np.linalg.norm(d.qvel[6:29]))])

    return frames, np.array(tel, np.float32), caido_desde


def guardar(cfg, frames, tel, caido_desde):
    nombre = cfg["nombre"]
    mp4 = os.path.join(OUT_MEDIA, nombre + ".mp4")
    gif = os.path.join(OUT_MEDIA, nombre + ".gif")

    imageio.mimsave(mp4, frames, fps=FPS, quality=8, macro_block_size=None)

    # GIF: 20 fps, 560 px de ancho
    paso = max(1, FPS // 20)
    peq = [np.asarray(Image.fromarray(f).resize((560, int(560 * H / W)), Image.LANCZOS))
           for f in frames[::paso]]
    imageio.mimsave(gif, peq, duration=1000 / 20, loop=0)

    n = len(frames)
    for etiqueta, idx in [("inicio", 0), ("medio", n // 2), ("final", n - 1)]:
        Image.fromarray(frames[idx]).save(
            os.path.join(OUT_FRAMES, f"{nombre}_{etiqueta}.png"))

    np.savez(os.path.join(OUT_DATA, nombre + ".npz"), tel=tel,
             titulo=cfg["titulo"], desc=cfg["desc"], archivo=cfg["archivo"],
             esperado=cfg.get("esperado", ""),
             caido=-1.0 if caido_desde is None else caido_desde)

    tam = lambda p: os.path.getsize(p) / 1e6
    res = "CAIDA en t=%.2fs" % caido_desde if caido_desde is not None else "ESTABLE toda la corrida"
    print(f"  -> {nombre}.mp4 ({tam(mp4):.1f} MB)  {nombre}.gif ({tam(gif):.1f} MB)  [{res}]")
    return caido_desde


# ---------------------------------------------------------------------------
# ESCENARIOS
# ---------------------------------------------------------------------------
ESCENARIOS = [
    dict(nombre="01_g1_pose_base_pd", controlador="pd", duracion=8.0,
         titulo="01 - G1 de pie: control PD puro",
         desc="play_amo_stable.py - PD hacia BASE_POSE, sin policy de IA",
         archivo="play_amo_stable.py", esperado="CAE ~2.8 s", brazos=[(0.0, "neutral")]),

    dict(nombre="02_g1_marcha_pd", controlador="pd", duracion=12.0,
         titulo="02 - G1 marcha con patron ciclico PD",
         desc="Comando vx=0.35 m/s, patron sinusoidal de cadera/rodilla/tobillo",
         archivo="play_amo_stable.py", esperado="CAE ~2.8 s", cam_giro=True,
         comandos=[(0.0, {}), (2.0, {"vx": 0.35}), (9.0, {"vx": 0.0})]),

    dict(nombre="03_g1_altura_torso", controlador="pd", duracion=10.0,
         titulo="03 - G1 control de altura de torso",
         desc="Comando de altura (tecla Z/X): flexion coordinada cadera-rodilla-tobillo",
         archivo="play_amo_stable.py", esperado="CAE ~2.8 s",
         comandos=[(0.0, {}), (2.0, {"height": -0.10}), (5.0, {"height": 0.15}),
                   (8.0, {"height": 0.0})]),

    dict(nombre="04_g1_brazos_secuencias", controlador="pd", duracion=16.0,
         titulo="04 - G1 secuencias de brazos (ArmController)",
         desc="Saludo, cruz, carga, guardia y apuntado sobre postura PD estable",
         archivo="ArmController.py + banda_v2.py", esperado="CAE ~2.8 s", cam_dist=2.7,
         brazos=[(0.0, "neutral"), (1.5, "saludo_a"), (3.0, "saludo_b"), (4.0, "saludo_a"),
                 (5.0, "saludo_b"), (6.5, "cruz"), (9.0, "carga"), (11.0, "guardia"),
                 (13.0, "apunta"), (15.0, "neutral")]),

    dict(nombre="05_g1_policy_amo", controlador="amo", duracion=8.0,
         titulo="05 - Policy AMO real (amo_jit.pt + adapter_jit.pt)",
         desc="Pipeline completo: adapter -> historial 11x93 -> policy -> 15 pares",
         archivo="play_amo.py", esperado="CAE 0.34 s",
         comandos=[(0.0, {}), (1.5, {"vx": 0.3})]),

    dict(nombre="06_fallo_sin_adapter", controlador="amo", duracion=8.0,
         titulo="06 - FALLO: adapter desconectado",
         desc="adapter_out forzado a cero: la policy pierde la estimacion de la base",
         archivo="play_amo.py (degradado)", esperado="FALLO",
         fallos={"adapter_off": True},
         comandos=[(0.0, {}), (1.5, {"vx": 0.3})]),

    dict(nombre="07_fallo_ruido_pares", controlador="pd", duracion=9.0,
         titulo="07 - FALLO: ruido en los pares de los actuadores",
         desc="Ruido gaussiano sigma=22 Nm sobre data.ctrl (emula driver/latencia sucia)",
         archivo="play_amo_stable.py (degradado)", esperado="FALLO",
         fallos={"ruido_par": 22.0}, semilla=3),

    dict(nombre="08_fallo_ganancias_bajas", controlador="pd", duracion=7.0,
         titulo="08 - FALLO: ganancias PD al 20%",
         desc="KP*0.20 y KD*0.20: los pares no sostienen el peso, colapso de rodillas",
         archivo="play_amo_stable.py (degradado)", esperado="FALLO",
         fallos={"kp_escala": 0.20, "kd_escala": 0.20}),

    dict(nombre="09_fallo_latencia", controlador="pd", duracion=8.0,
         titulo="09 - FALLO: 45 ms de latencia en el estado",
         desc="Realimentacion retrasada: el lazo PD entra en oscilacion divergente",
         archivo="play_amo_stable.py (degradado)", esperado="FALLO",
         fallos={"latencia_ms": 45}),

    dict(nombre="10_fallo_pose_inicial", controlador="pd", duracion=7.0,
         titulo="10 - FALLO: pose inicial sin flexion (qpos=0)",
         desc="El problema de estabilidad inicial de banda.py: piernas rectas al arrancar",
         archivo="banda.py", esperado="FALLO",
         fallos={"pose_inicial": "cero"}),

    dict(nombre="11_fallo_arranque_inclinado", controlador="pd", duracion=7.0,
         titulo="11 - FALLO: arranque con 16 grados de pitch",
         desc="Perturbacion de orientacion inicial: el PD no recupera el equilibrio",
         archivo="play_amo_stable.py (degradado)", esperado="FALLO",
         fallos={"pose_inicial": "inclinado"}),

    dict(nombre="12_fallo_caida_libre", controlador="pd", duracion=6.0,
         titulo="12 - FALLO: soltado desde 1.15 m",
         desc="Impacto de aterrizaje: los tobillos saturan y el robot rebota y cae",
         archivo="play_amo_stable.py (degradado)", esperado="FALLO",
         fallos={"pose_inicial": "alto"}),

    dict(nombre="13_fallo_escala_accion", controlador="amo", duracion=7.0,
         titulo="13 - FALLO: escala de accion x3",
         desc="Salida de la policy multiplicada por 3: pares excesivos, robot descontrolado",
         archivo="play_amo.py (degradado)", esperado="FALLO",
         fallos={"escala_accion": 3.0},
         comandos=[(0.0, {}), (1.0, {"vx": 0.3})]),
    dict(nombre="14_g1_brazos_pelvis_fija", controlador="pd", duracion=16.0,
         titulo="14 - Poses de brazos con pelvis fijada",
         desc="Banco de pruebas: pelvis anclada para validar las 7 poses del ArmController",
         archivo="ArmController.py", esperado="BANCO DE PRUEBAS", cam_dist=2.5, cam_giro=True,
         pelvis_fija=True,
         brazos=[(0.0, "neutral"), (1.5, "saludo_a"), (3.0, "saludo_b"), (4.0, "saludo_a"),
                 (5.0, "saludo_b"), (6.5, "cruz"), (9.0, "carga"), (11.0, "guardia"),
                 (13.0, "apunta"), (15.0, "neutral")]),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--lista" in sys.argv:
        for c in ESCENARIOS:
            print(f"{c['nombre']:32s} {c.get('esperado',''):18s} {c['desc']}")
        return

    sel = [c for c in ESCENARIOS if not args or any(c["nombre"].startswith(a) for a in args)]
    model = load_model()
    renderer = mujoco.Renderer(model, H, W)
    print(f"[render_g1] {len(sel)} escenario(s), {W}x{H} @ {FPS} fps\n")

    resumen = []
    for c in sel:
        print(f"[{c['nombre']}] {c['titulo']}")
        frames, tel, caido = correr(c, model, renderer)
        guardar(c, frames, tel, caido)
        resumen.append((c["nombre"], c.get("esperado", ""), caido))

    print("\n== RESUMEN ==")
    for n, esp, caido in resumen:
        est = "ESTABLE" if caido is None else f"CAIDA t={caido:.2f}s"
        print(f"  {n:32s} esperado={esp:18s} resultado={est}")


if __name__ == "__main__":
    main()
