// r1_deploy.cpp
// ---------------------------------------------------------------------------
// Despliegue de la politica NATIVA de Isaac en el R1 usando LibTorch +
// el wrapper real de unitree_sdk2 (confirmado contra los headers instalados
// en /usr/local/include/unitree/dds_wrapper/robots/g1/{g1_pub,g1_sub}.h).
//
// CONFIRMADO (no es una suposicion):
//   - El R1 tiene 24 actuadores en r1.xml -> 24 > NUM_MOTOR_IDL_GO(20) ->
//     unitree_mujoco lo sirve por G1Bridge -> IDL unitree_hg.
//   - API real: LowCmd_t::trylock()/msg_/unlockAndPublish() (CRC automatico
//     via pre_communication()). LowState_t::wait_for_connection()/msg_
//     protegido por mutex_ publico.
//
// >>> MODOS DE USO (cambia SOLO estas 2 constantes) <<<
//   Validacion en unitree_mujoco (loopback):  DOMAIN_ID=1, NET_IFACE="lo"
//   Robot real (cuando llegue el momento):    DOMAIN_ID=0 (o el que uses),
//                                              NET_IFACE = tu interfaz real
//
// >>> LO QUE SIGUE PENDIENTE PARA EL ROBOT REAL (no aplica a la sim) <<<
//   1. SDK_INDEX[]  -> aqui es identidad porque unitree_mujoco respeta el
//      orden del XML tal cual; en el robot real puede que el firmware
//      numere los motores distinto -> VERIFICAR moviendo motor por motor.
//   2. Convencion de la IMU (frame, montaje) en el robot real.
//   3. mode() del motor_cmd: en sim no importa (el bridge lo ignora), en
//      el robot real hay que confirmar que valor espera el firmware.
// ---------------------------------------------------------------------------

#include <torch/script.h>

#include <array>
#include <deque>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <atomic>
#include <csignal>
#include <mutex>

// ===========================================================================
// INCLUDES DEL SDK  (confirmados contra los headers instalados)
// ===========================================================================
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/robots/g1/g1.h>

using LowCmd_t   = unitree::robot::g1::publisher::LowCmd;      // trylock/msg_/unlockAndPublish
using LowState_t = unitree::robot::g1::subscription::LowState; // msg_ + mutex_, wait_for_connection()

// --- Modo de conexion: cambia estas 2 lineas segun sim/real ---
static constexpr int  DOMAIN_ID = 1;      // unitree_mujoco/simulate/config.yaml -> domain_id: 1
static const char*    NET_IFACE = "lo";   // unitree_mujoco/simulate/config.yaml -> interface: "lo"

// ===========================================================================
// CONSTANTES  (portadas tal cual del script de Python)
// ===========================================================================
static constexpr int NUM_DOFS    = 24;   // <-- TODO 4: confirmar contra tu LowState_ real
static constexpr int HISTORY_LEN  = 5;
static constexpr int OBS_DIM      = 405;

static constexpr double CONTROL_HZ = 50.0;
static constexpr double CONTROL_DT = 1.0 / CONTROL_HZ;   // 0.02 s
static constexpr float  ACTION_SCALE    = 0.25f;
static constexpr float  SCALE_ANG_VEL   = 0.2f;
static constexpr float  SCALE_JOINT_VEL = 0.05f;

// Array de oro: target_mj[i] = target_isaac[ MUJOCO_FROM_ISAAC[i] ]
static const std::array<int, NUM_DOFS> MUJOCO_FROM_ISAAC = {
    0, 3, 6, 10, 14, 18,  1, 4, 7, 11, 15, 19,
    2, 5, 8, 12, 16, 20, 22,  9, 13, 17, 21, 23
};

// Pose por defecto en ORDEN DE DESPLIEGUE (== orden MuJoCo del CFG)
static const std::array<float, NUM_DOFS> DEFAULT_MJ = {
    -0.06f, 0.0f, 0.0f, 0.05f, -0.04f, 0.0f,
    -0.06f, 0.0f, 0.0f, 0.05f, -0.04f, 0.0f,
     0.0f,  0.0f,
     0.18f, 0.18f, 0.0f, 1.5f, 0.0f,
     0.18f,-0.18f, 0.0f, 1.5f, 0.0f
};

static const std::array<float, NUM_DOFS> STIFFNESS_MJ = {
    100,100,100,200,80,80,  100,100,100,200,80,80,
    250,250,  50,50,40,30,20,  50,50,40,30,20
};
static const std::array<float, NUM_DOFS> DAMPING_MJ = {
    5,5,5,8,5,5,  5,5,5,8,5,5,
    25,25,  5,5,4,4,4,  5,5,4,4,4
};

// TODO (solo robot real): reemplaza con limites reales por joint (rad),
// sacados de tu URDF. El margen simetrico de abajo es solo un placeholder
// de seguridad -- para la validacion en unitree_mujoco no hay riesgo fisico.
static const float SAFE_MARGIN = 1.5f;  // rad alrededor del default

// Identidad: unitree_mujoco (G1Bridge) copia motor_state()[i] <-> actuator i
// del XML sin reordenar. Como r1.xml ya esta en el mismo orden que usa nuestra
// politica (pierna izq, pierna der, cintura, brazo izq, brazo der), no hace
// falta ningun remapeo para la simulacion. Para el robot real, VERIFICAR.
static std::array<int, NUM_DOFS> SDK_INDEX = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
};

static std::atomic<bool> g_stop{false};
static void onSigint(int) { g_stop.store(true); }

// ===========================================================================
// UTILIDADES
// ===========================================================================
static void quatRotateInverse(const float q[4], const float v[3], float out[3]) {
    const float w = q[0];
    const float qv[3] = {q[1], q[2], q[3]};
    const float dot = qv[0]*v[0] + qv[1]*v[1] + qv[2]*v[2];
    const float cross[3] = {
        qv[1]*v[2] - qv[2]*v[1],
        qv[2]*v[0] - qv[0]*v[2],
        qv[0]*v[1] - qv[1]*v[0]
    };
    for (int i = 0; i < 3; ++i)
        out[i] = v[i]*(2.0f*w*w - 1.0f) - cross[i]*(2.0f*w) + qv[i]*(2.0f*dot);
}

static void sleepUntil(struct timespec& next) {
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
    long ns = next.tv_nsec + (long)(CONTROL_DT * 1e9);
    next.tv_sec  += ns / 1000000000L;
    next.tv_nsec  = ns % 1000000000L;
}

// ===========================================================================
// TIPOS DE INTERCAMBIO  (ORDEN DE DESPLIEGUE)
// ===========================================================================
struct RobotState {
    std::array<float, NUM_DOFS> joint_pos{};
    std::array<float, NUM_DOFS> joint_vel{};
    std::array<float, 3>        ang_vel{};
    std::array<float, 4>        quat{1,0,0,0};
};

struct MotorCmd {
    float q, dq, kp, kd, tau;
};

// ===========================================================================
// CAPA DEL SDK  -- usa el wrapper real (LowCmd_t/LowState_t), no DDS crudo.
// El CRC lo calcula solo LowCmd_t::pre_communication() al hacer
// unlockAndPublish(); no hay que tocarlo.
// ===========================================================================
class RobotSDK {
public:
    void init() {
        unitree::robot::ChannelFactory::Instance()->Init(DOMAIN_ID, NET_IFACE);

        lowcmd_   = std::make_unique<LowCmd_t>();
        lowstate_ = std::make_shared<LowState_t>();

        printf("Esperando conexion a rt/lowstate (domain=%d, iface=%s)...\n",
               DOMAIN_ID, NET_IFACE);
        lowstate_->wait_for_connection();  // bloquea hasta el primer mensaje
        printf("Conectado.\n");

        // TODO (solo robot real): aqui va el equivalente a apagar el
        // servicio de movimiento de alto nivel si el R1 tiene uno (como
        // MotionSwitcherClient en G1). Para unitree_mujoco no aplica: el
        // simulador nace en modo "solo escucha low-level", no hay FSM de
        // alto nivel que pelee con nosotros.
    }

    RobotState readState() {
        RobotState s{};
        std::lock_guard<std::mutex> lk(lowstate_->mutex_);
        for (int i = 0; i < NUM_DOFS; ++i) {
            int m = SDK_INDEX[i];
            s.joint_pos[i] = lowstate_->msg_.motor_state()[m].q();
            s.joint_vel[i] = lowstate_->msg_.motor_state()[m].dq();
        }
        const auto& imu = lowstate_->msg_.imu_state();
        s.ang_vel = { imu.gyroscope()[0], imu.gyroscope()[1], imu.gyroscope()[2] };
        s.quat    = { imu.quaternion()[0], imu.quaternion()[1],
                      imu.quaternion()[2], imu.quaternion()[3] };
        // TODO (solo robot real): si la IMU esta rotada respecto a la base,
        // rota s.ang_vel y s.quat aqui. En unitree_mujoco la IMU del XML ya
        // esta alineada con la base, asi que para la validacion no aplica.
        return s;
    }

    void sendCommand(const std::array<MotorCmd, NUM_DOFS>& cmd) {
        // trylock() puede fallar si el hilo de publicacion interno tiene el
        // lock en ese instante exacto (ventana de microsegundos). Reintenta
        // brevemente en vez de perder el tick de control en silencio.
        int attempts = 0;
        while (!lowcmd_->trylock()) {
            if (++attempts > 50) {  // ~5 ms de reintentos a 100us cada uno
                fprintf(stderr, "WARN: no se pudo lockear lowcmd, se perdio un tick\n");
                return;
            }
            usleep(100);
        }
        for (int i = 0; i < NUM_DOFS; ++i) {
            int m = SDK_INDEX[i];
            auto& mc = lowcmd_->msg_.motor_cmd()[m];
            mc.q()    = cmd[i].q;
            mc.dq()   = cmd[i].dq;
            mc.kp()   = cmd[i].kp;
            mc.kd()   = cmd[i].kd;
            mc.tau()  = cmd[i].tau;
            mc.mode() = 1;  // el bridge de unitree_mujoco lo ignora; para el
                             // robot real hay que confirmar el valor esperado.
        }
        lowcmd_->unlockAndPublish();  // calcula CRC y publica
    }

private:
    std::unique_ptr<LowCmd_t>   lowcmd_;
    std::shared_ptr<LowState_t> lowstate_;
};

// ===========================================================================
// POLITICA  (LibTorch) -- SIN CAMBIOS respecto a la version anterior
// ===========================================================================
class PolicyRunner {
public:
    explicit PolicyRunner(const std::string& path) {
        module_ = torch::jit::load(path);
        module_.eval();

        for (int isaac = 0; isaac < NUM_DOFS; ++isaac)
            isaac_from_mujoco_[MUJOCO_FROM_ISAAC[isaac]] = isaac;

        std::array<float, NUM_DOFS> tmp;
        mjToIsaac(DEFAULT_MJ, tmp);
        default_isaac_ = tmp;

        reset();
    }

    void reset() {
        last_action_isaac_.fill(0.0f);
        for (auto* dq : {&h_ang_vel_, &h_grav_, &h_cmd_})
            *dq = std::deque<std::array<float,3>>(HISTORY_LEN, std::array<float,3>{});
        for (auto* dq : {&h_jpos_, &h_jvel_, &h_act_})
            *dq = std::deque<std::array<float,NUM_DOFS>>(HISTORY_LEN, std::array<float,NUM_DOFS>{});
    }

    std::array<float, NUM_DOFS> step(const RobotState& st,
                                     const std::array<float,3>& command) {
        float obs[OBS_DIM];
        fillObs(st, command, obs);

        torch::Tensor in = torch::from_blob(obs, {1, OBS_DIM}, torch::kFloat32).clone();
        torch::NoGradGuard ng;
        torch::Tensor out = module_.forward({in}).toTensor().to(torch::kCPU).contiguous();
        const float* a = out.data_ptr<float>();

        std::array<float, NUM_DOFS> action_isaac, target_isaac, target_mj;
        for (int i = 0; i < NUM_DOFS; ++i) action_isaac[i] = a[i];
        last_action_isaac_ = action_isaac;

        for (int i = 0; i < NUM_DOFS; ++i)
            target_isaac[i] = default_isaac_[i] + ACTION_SCALE * action_isaac[i];
        isaacToMj(target_isaac, target_mj);
        return target_mj;
    }

private:
    void mjToIsaac(const std::array<float,NUM_DOFS>& mj,
                   std::array<float,NUM_DOFS>& isaac) const {
        for (int i = 0; i < NUM_DOFS; ++i) isaac[i] = mj[isaac_from_mujoco_[i]];
    }
    void isaacToMj(const std::array<float,NUM_DOFS>& isaac,
                   std::array<float,NUM_DOFS>& mj) const {
        for (int i = 0; i < NUM_DOFS; ++i) mj[i] = isaac[MUJOCO_FROM_ISAAC[i]];
    }

    void fillObs(const RobotState& st, const std::array<float,3>& command, float* obs) {
        std::array<float,3> ang_vel_obs, grav_obs;
        for (int i = 0; i < 3; ++i) ang_vel_obs[i] = st.ang_vel[i] * SCALE_ANG_VEL;

        const float g_world[3] = {0.0f, 0.0f, -1.0f};
        float g_body[3];
        quatRotateInverse(st.quat.data(), g_world, g_body);
        for (int i = 0; i < 3; ++i) grav_obs[i] = g_body[i];

        std::array<float,NUM_DOFS> jpos_isaac, jvel_isaac, tmp;
        mjToIsaac(st.joint_pos, tmp);
        for (int i = 0; i < NUM_DOFS; ++i) jpos_isaac[i] = tmp[i] - default_isaac_[i];
        mjToIsaac(st.joint_vel, tmp);
        for (int i = 0; i < NUM_DOFS; ++i) jvel_isaac[i] = tmp[i] * SCALE_JOINT_VEL;

        auto push3 = [](std::deque<std::array<float,3>>& d, const std::array<float,3>& v){
            d.push_back(v); d.pop_front();
        };
        auto pushN = [](std::deque<std::array<float,NUM_DOFS>>& d, const std::array<float,NUM_DOFS>& v){
            d.push_back(v); d.pop_front();
        };
        push3(h_ang_vel_, ang_vel_obs);
        push3(h_grav_,    grav_obs);
        push3(h_cmd_,     command);
        pushN(h_jpos_,    jpos_isaac);
        pushN(h_jvel_,    jvel_isaac);
        pushN(h_act_,     last_action_isaac_);

        int k = 0;
        for (auto& s : h_ang_vel_) for (float x : s) obs[k++] = x;
        for (auto& s : h_grav_)    for (float x : s) obs[k++] = x;
        for (auto& s : h_cmd_)     for (float x : s) obs[k++] = x;
        for (auto& s : h_jpos_)    for (float x : s) obs[k++] = x;
        for (auto& s : h_jvel_)    for (float x : s) obs[k++] = x;
        for (auto& s : h_act_)     for (float x : s) obs[k++] = x;
    }

    torch::jit::script::Module module_;
    std::array<int,   NUM_DOFS> isaac_from_mujoco_{};
    std::array<float, NUM_DOFS> default_isaac_{};
    std::array<float, NUM_DOFS> last_action_isaac_{};

    std::deque<std::array<float,3>>        h_ang_vel_, h_grav_, h_cmd_;
    std::deque<std::array<float,NUM_DOFS>> h_jpos_, h_jvel_, h_act_;
};

// ===========================================================================
// ARRANQUE SEGURO
// ===========================================================================
static void moveToDefault(RobotSDK& sdk, double seconds = 3.0) {
    RobotState st = sdk.readState();
    std::array<float, NUM_DOFS> start = st.joint_pos;

    const int steps = (int)(seconds * CONTROL_HZ);
    struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next);

    for (int s = 0; s <= steps && !g_stop.load(); ++s) {
        const float alpha = (float)s / steps;
        std::array<MotorCmd, NUM_DOFS> cmd;
        for (int i = 0; i < NUM_DOFS; ++i) {
            float q = (1.0f - alpha) * start[i] + alpha * DEFAULT_MJ[i];
            // rampa de ganancias tambien, para no pegar un salto de torque
            // si el motor ya no estaba en modo posicion.
            float kp = alpha * STIFFNESS_MJ[i];
            float kd = alpha * DAMPING_MJ[i];
            cmd[i] = { q, 0.0f, kp, kd, 0.0f };
        }
        sdk.sendCommand(cmd);
        sleepUntil(next);
    }
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char** argv) {
    const std::string policy_path = (argc > 1) ? argv[1] : "policies/r1_policy_v2.pt";
    std::signal(SIGINT, onSigint);

    RobotSDK sdk;
    sdk.init();

    printf("Cargando politica: %s\n", policy_path.c_str());
    PolicyRunner policy(policy_path);

    printf("Moviendo a pose default (mantente listo para el paro, robot suspendido)...\n");
    moveToDefault(sdk, 3.0);

    std::array<float, 3> command = {0.0f, 0.0f, 0.0f};
    // TODO: engancha aqui una fuente real de comando (joystick/planner) en
    // vez de dejarlo fijo en cero. Ojo: cambiar `command` desde otro hilo
    // requiere sincronizacion (mutex/atomic), como con state_mtx_ arriba.

    printf("== Politica ACTIVA (50 Hz) -- Ctrl+C para detener ==\n");
    struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next);

    while (!g_stop.load()) {
        RobotState st = sdk.readState();

        std::array<float, NUM_DOFS> target_mj = policy.step(st, command);

        std::array<MotorCmd, NUM_DOFS> cmd;
        for (int i = 0; i < NUM_DOFS; ++i) {
            float lo = DEFAULT_MJ[i] - SAFE_MARGIN;
            float hi = DEFAULT_MJ[i] + SAFE_MARGIN;
            float q  = std::min(std::max(target_mj[i], lo), hi);
            cmd[i] = { q, 0.0f, STIFFNESS_MJ[i], DAMPING_MJ[i], 0.0f };
        }
        sdk.sendCommand(cmd);

        sleepUntil(next);
    }

    // Al salir: baja ganancias suavemente en vez de cortar en seco.
    printf("\nDeteniendo: relajando ganancias...\n");
    RobotState st = sdk.readState();
    std::array<float, NUM_DOFS> hold = st.joint_pos;
    const int relax_steps = (int)(1.0 * CONTROL_HZ);
    for (int s = 0; s <= relax_steps; ++s) {
        float alpha = 1.0f - (float)s / relax_steps;
        std::array<MotorCmd, NUM_DOFS> cmd;
        for (int i = 0; i < NUM_DOFS; ++i)
            cmd[i] = { hold[i], 0.0f, alpha * STIFFNESS_MJ[i], alpha * DAMPING_MJ[i], 0.0f };
        sdk.sendCommand(cmd);
        sleepUntil(next);
    }
    printf("Fin\n");
    return 0;
}