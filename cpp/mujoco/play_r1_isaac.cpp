// play_r1_isaac.cpp
// ---------------------------------------------------------------------------
// Puerto DIRECTO a C++ de play_r1_isaac.py: misma logica exacta (obs 405,
// history de 5 pasos, reordenamiento Isaac<->MuJoCo, PD local con
// TORQUE_LIMITS_MJ), pero usando la API C de MuJoCo + GLFW en vez de
// mujoco.viewer de Python. NO usa DDS ni unitree_sdk2 -- es standalone.
//
// Controles (identicos al script de Python):
//   flechas  : vx (arriba/abajo), wz/giro (izq/der)
//   Q / E    : vy lateral
//   ESPACIO  : detener (comando = 0)
//   R        : reset a la pose inicial
//   ESC      : salir
//   mouse    : rotar/paneo/zoom de camara (arrastrar + rueda)
//
// Uso:
//   ./play_r1_isaac <ruta_r1.xml> <ruta_r1_policy_v2.pt>
// ---------------------------------------------------------------------------

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <torch/script.h>

#include <array>
#include <deque>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <string>

// ===========================================================================
// CONSTANTES  (copiadas 1:1 de play_r1_isaac.py)
// ===========================================================================
static constexpr int NUM_DOFS    = 24;
static constexpr int HISTORY_LEN = 5;
static constexpr int OBS_DIM     = 405;

static constexpr double SIM_DT      = 0.002;  // paso de fisica
static constexpr int    DECIMATION  = 10;     // 0.002*10 = 0.02s -> 50Hz de politica
static constexpr float  ACTION_SCALE    = 0.25f;
static constexpr float  SCALE_ANG_VEL   = 0.2f;
static constexpr float  SCALE_JOINT_VEL = 0.05f;

// accion_mujoco = accion_isaac[MUJOCO_FROM_ISAAC]  (el "array de oro")
static const std::array<int, NUM_DOFS> MUJOCO_FROM_ISAAC = {
    0, 3, 6, 10, 14, 18,  1, 4, 7, 11, 15, 19,
    2, 5, 8, 12, 16, 20, 22,  9, 13, 17, 21, 23
};
static std::array<int, NUM_DOFS> ISAAC_FROM_MUJOCO{}; // se calcula en main() (argsort)

// Pose por defecto en ORDEN MUJOCO
static const std::array<float, NUM_DOFS> DEFAULT_MJ = {
    -0.06f, 0.0f, 0.0f, 0.05f, -0.04f, 0.0f,
    -0.06f, 0.0f, 0.0f, 0.05f, -0.04f, 0.0f,
     0.0f,  0.0f,
     0.18f, 0.18f, 0.0f, 1.5f, 0.0f,
     0.18f,-0.18f, 0.0f, 1.5f, 0.0f
};
static std::array<float, NUM_DOFS> DEFAULT_ISAAC{}; // se calcula en main()

static const std::array<float, NUM_DOFS> STIFFNESS_MJ = {
    100,100,100,200,80,80,  100,100,100,200,80,80,
    250,250,  50,50,40,30,20,  50,50,40,30,20
};
static const std::array<float, NUM_DOFS> DAMPING_MJ = {
    5,5,5,8,5,5,  5,5,5,8,5,5,
    25,25,  5,5,4,4,4,  5,5,4,4,4
};
static const std::array<float, NUM_DOFS> TORQUE_LIMITS_MJ = {
    88,139,88,139,50,50,  88,139,88,139,50,50,
    88,50,  25,25,25,25,25,  25,25,25,25,25
};

// ===========================================================================
// UTILIDADES
// ===========================================================================
static void mjToIsaac(const std::array<float,NUM_DOFS>& mj, std::array<float,NUM_DOFS>& isaac) {
    for (int i = 0; i < NUM_DOFS; ++i) isaac[i] = mj[ISAAC_FROM_MUJOCO[i]];
}
static void isaacToMj(const std::array<float,NUM_DOFS>& isaac, std::array<float,NUM_DOFS>& mj) {
    for (int i = 0; i < NUM_DOFS; ++i) mj[i] = isaac[MUJOCO_FROM_ISAAC[i]];
}

// Equivalente a quat_rotate_inverse() de Python: rota v del frame mundo al
// frame del cuerpo. q en orden (w,x,y,z).
static void quatRotateInverse(const double q[4], const float v[3], float out[3]) {
    const float w = (float)q[0];
    const float qv[3] = { (float)q[1], (float)q[2], (float)q[3] };
    const float dot = qv[0]*v[0] + qv[1]*v[1] + qv[2]*v[2];
    const float cross[3] = {
        qv[1]*v[2] - qv[2]*v[1],
        qv[2]*v[0] - qv[0]*v[2],
        qv[0]*v[1] - qv[1]*v[0]
    };
    for (int i = 0; i < 3; ++i)
        out[i] = v[i]*(2.0f*w*w - 1.0f) - cross[i]*(2.0f*w) + qv[i]*(2.0f*dot);
}

// ===========================================================================
// ESTADO GLOBAL  (patron estandar de los ejemplos de MuJoCo: simple.cc/basic.c)
// ===========================================================================
static mjModel* m = nullptr;
static mjData*  d = nullptr;

static mjvCamera cam;
static mjvOption opt;
static mjvScene  scn;
static mjrContext con;

// mouse
static bool button_left = false, button_middle = false, button_right = false;
static double lastx = 0, lasty = 0;

// politica
static torch::jit::script::Module policy;

// comando [vx, vy, wz]
static std::array<float, 3> command = {0.f, 0.f, 0.f};

// last_action en ORDEN ISAAC (salida cruda de la red, paso anterior)
static std::array<float, NUM_DOFS> last_action_isaac{};

// history (deques de snapshots ya escalados, orden Isaac)
static std::deque<std::array<float,3>>        h_ang_vel, h_grav, h_cmd;
static std::deque<std::array<float,NUM_DOFS>> h_jpos, h_jvel, h_act;

static bool has_gyro = false;
static int  gyro_adr = -1;

// ===========================================================================
// RESET  (equivalente a R1IsaacPolicyEnv.reset())
// ===========================================================================
static void reset_sim() {
    mj_resetData(m, d);
    d->qpos[0] = 0.0; d->qpos[1] = 0.0; d->qpos[2] = 0.74;
    d->qpos[3] = 1.0; d->qpos[4] = 0.0; d->qpos[5] = 0.0; d->qpos[6] = 0.0;
    for (int i = 0; i < NUM_DOFS; ++i) d->qpos[7 + i] = DEFAULT_MJ[i];
    for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
    mj_forward(m, d);

    last_action_isaac.fill(0.0f);
    h_ang_vel.assign(HISTORY_LEN, std::array<float,3>{});
    h_grav.assign(HISTORY_LEN, std::array<float,3>{});
    h_cmd.assign(HISTORY_LEN, std::array<float,3>{});
    h_jpos.assign(HISTORY_LEN, std::array<float,NUM_DOFS>{});
    h_jvel.assign(HISTORY_LEN, std::array<float,NUM_DOFS>{});
    h_act.assign(HISTORY_LEN, std::array<float,NUM_DOFS>{});
}

// ===========================================================================
// OBSERVACION  (equivalente a _build_obs())
// ===========================================================================
static void buildObs(float* obs /* [OBS_DIM] */) {
    std::array<float, NUM_DOFS> qpos_mj, qvel_mj;
    for (int i = 0; i < NUM_DOFS; ++i) {
        qpos_mj[i] = (float)d->qpos[7 + i];
        qvel_mj[i] = (float)d->qvel[6 + i];
    }
    const double* quat = &d->qpos[3]; // (w,x,y,z)

    std::array<float,3> ang_vel{};
    if (has_gyro) {
        ang_vel = { (float)d->sensordata[gyro_adr + 0],
                    (float)d->sensordata[gyro_adr + 1],
                    (float)d->sensordata[gyro_adr + 2] };
    } else {
        ang_vel = { (float)d->qvel[3], (float)d->qvel[4], (float)d->qvel[5] };
    }

    std::array<float,3> ang_vel_obs, grav_obs;
    for (int i = 0; i < 3; ++i) ang_vel_obs[i] = ang_vel[i] * SCALE_ANG_VEL;
    const float g_world[3] = {0.f, 0.f, -1.f};
    quatRotateInverse(quat, g_world, grav_obs.data());

    std::array<float,3> cmd_obs = command;

    std::array<float,NUM_DOFS> jpos_isaac, jvel_isaac, tmp;
    mjToIsaac(qpos_mj, tmp);
    for (int i = 0; i < NUM_DOFS; ++i) jpos_isaac[i] = tmp[i] - DEFAULT_ISAAC[i];
    mjToIsaac(qvel_mj, tmp);
    for (int i = 0; i < NUM_DOFS; ++i) jvel_isaac[i] = tmp[i] * SCALE_JOINT_VEL;

    h_ang_vel.push_back(ang_vel_obs); h_ang_vel.pop_front();
    h_grav.push_back(grav_obs);       h_grav.pop_front();
    h_cmd.push_back(cmd_obs);         h_cmd.pop_front();
    h_jpos.push_back(jpos_isaac);     h_jpos.pop_front();
    h_jvel.push_back(jvel_isaac);     h_jvel.pop_front();
    h_act.push_back(last_action_isaac); h_act.pop_front();

    int k = 0;
    for (auto& s : h_ang_vel) for (float x : s) obs[k++] = x; // 15
    for (auto& s : h_grav)    for (float x : s) obs[k++] = x; // 15
    for (auto& s : h_cmd)     for (float x : s) obs[k++] = x; // 15
    for (auto& s : h_jpos)    for (float x : s) obs[k++] = x; // 120
    for (auto& s : h_jvel)    for (float x : s) obs[k++] = x; // 120
    for (auto& s : h_act)     for (float x : s) obs[k++] = x; // 120
    // k == 405
}

// ===========================================================================
// PASO DE POLITICA  (equivalente a _policy_step())
// ===========================================================================
static std::array<float, NUM_DOFS> policyStep() {
    float obs[OBS_DIM];
    buildObs(obs);

    torch::Tensor obs_t = torch::from_blob(obs, {1, OBS_DIM}, torch::kFloat32).clone();
    torch::NoGradGuard ng;
    torch::Tensor out = policy.forward({obs_t}).toTensor().to(torch::kCPU).contiguous();
    const float* a = out.data_ptr<float>();

    std::array<float, NUM_DOFS> action_isaac, target_isaac, target_mj;
    for (int i = 0; i < NUM_DOFS; ++i) action_isaac[i] = a[i];
    last_action_isaac = action_isaac;

    for (int i = 0; i < NUM_DOFS; ++i)
        target_isaac[i] = DEFAULT_ISAAC[i] + ACTION_SCALE * action_isaac[i];
    isaacToMj(target_isaac, target_mj);
    return target_mj;
}

// ===========================================================================
// PD LOCAL con clamp de torque  (equivalente a _apply_pd())
// ===========================================================================
static void applyPd(const std::array<float, NUM_DOFS>& target_mj) {
    for (int i = 0; i < NUM_DOFS; ++i) {
        float q  = (float)d->qpos[7 + i];
        float qd = (float)d->qvel[6 + i];
        float torque = STIFFNESS_MJ[i] * (target_mj[i] - q) - DAMPING_MJ[i] * qd;
        torque = std::clamp(torque, -TORQUE_LIMITS_MJ[i], TORQUE_LIMITS_MJ[i]);
        d->ctrl[i] = torque;
    }
}

// ===========================================================================
// CALLBACKS DE GLFW  (teclado identico al key_cb de Python + mouse estandar
// de los ejemplos oficiales de MuJoCo para rotar/paneo/zoom de camara)
// ===========================================================================
static void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act != GLFW_PRESS) return; // solo en el flanco de bajada, como Python

    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); return;
        case GLFW_KEY_UP:     command[0] += 0.1f; break;
        case GLFW_KEY_DOWN:   command[0] -= 0.1f; break;
        case GLFW_KEY_LEFT:   command[2] += 0.1f; break;
        case GLFW_KEY_RIGHT:  command[2] -= 0.1f; break;
        case GLFW_KEY_Q:      command[1] += 0.1f; break;
        case GLFW_KEY_E:      command[1] -= 0.1f; break;
        case GLFW_KEY_SPACE:  command.fill(0.0f); break;
        case GLFW_KEY_R:      reset_sim(); return;
        default: return;
    }
    for (auto& c : command) c = std::clamp(c, -1.0f, 1.0f);
    printf("\rComando  vx=%+.2f  vy=%+.2f  wz=%+.2f   ", command[0], command[1], command[2]);
    fflush(stdout);
}

static void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    button_left   = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
    button_middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    button_right  = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
    glfwGetCursorPos(window, &lastx, &lasty);
}

static void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    if (!button_left && !button_middle && !button_right) return;
    double dx = xpos - lastx, dy = ypos - lasty;
    lastx = xpos; lasty = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
                      (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

static void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char** argv) {
    std::string xml_path    = (argc > 1) ? argv[1] : "assets/r1.xml";
    std::string policy_path = (argc > 2) ? argv[2] : "policies/r1_policy_v2.pt";

    // inverso del array de oro
    for (int isaac = 0; isaac < NUM_DOFS; ++isaac)
        ISAAC_FROM_MUJOCO[MUJOCO_FROM_ISAAC[isaac]] = isaac;
    mjToIsaac(DEFAULT_MJ, DEFAULT_ISAAC);

    // --- cargar modelo ---
    char error[1000] = "";
    printf("Cargando modelo: %s\n", xml_path.c_str());
    m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    if (!m) {
        fprintf(stderr, "ERROR cargando XML: %s\n", error);
        return 1;
    }
    m->opt.timestep = SIM_DT;
    d = mj_makeData(m);

    int gyro_id = mj_name2id(m, mjOBJ_SENSOR, "imu_ang_vel");
    if (gyro_id >= 0) {
        has_gyro = true;
        gyro_adr = m->sensor_adr[gyro_id];
    }

    // --- cargar politica ---
    printf("Cargando politica: %s\n", policy_path.c_str());
    policy = torch::jit::load(policy_path);
    policy.eval();

    reset_sim();

    // --- GLFW + visor MuJoCo (patron estandar simple.cc/basic.c) ---
    if (!glfwInit()) { fprintf(stderr, "No se pudo iniciar GLFW\n"); return 1; }
    GLFWwindow* window = glfwCreateWindow(1200, 900, "R1 - politica nativa Isaac (C++)", nullptr, nullptr);
    if (!window) { fprintf(stderr, "No se pudo crear la ventana\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    cam.distance  = 3.0;
    cam.elevation = -20;
    cam.azimuth   = 180;

    glfwSetKeyCallback(window, keyboard);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetScrollCallback(window, scroll);

    printf("\n== R1 con politica NATIVA de Isaac (C++/MuJoCo directo) ==\n");
    printf("Flechas: vx/giro | Q/E: lateral | ESPACIO: parar | R: reset | ESC: salir\n\n");

    std::array<float, NUM_DOFS> target_mj;
    for (int i = 0; i < NUM_DOFS; ++i) target_mj[i] = DEFAULT_MJ[i];
    long step = 0;

    while (!glfwWindowShouldClose(window)) {
        auto t0 = std::chrono::steady_clock::now();

        if (step % DECIMATION == 0) target_mj = policyStep();
        applyPd(target_mj);
        mj_step(m, d);

        cam.lookat[0] = d->qpos[0];
        cam.lookat[1] = d->qpos[1];
        cam.lookat[2] = d->qpos[2];

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        mjrRect viewport = {0, 0, width, height};
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();

        ++step;
        auto elapsed = std::chrono::steady_clock::now() - t0;
        auto target_dur = std::chrono::duration<double>(SIM_DT);
        auto sleep_dur = target_dur - elapsed;
        if (sleep_dur > std::chrono::duration<double>(0))
            std::this_thread::sleep_for(sleep_dur);
    }

    printf("\nFin\n");
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);
    glfwTerminate();
    return 0;
}