// teleop_r1.cpp
// ---------------------------------------------------------------------------
// Mitad "control" del teleop hibrido -- puerto directo de teleop_r1_v5.py
// EXCEPTO la parte de camara/MediaPipe, que se queda en pose_sender.py y
// llega aca por UDP (ver el comentario de formato de paquete en ese archivo).
//
// Reproduce: ArmIK (Gauss-Newton numerico), gate por confianza de
// profundidad, suavizado EMA + limite de paso, modo cinematico (fijo,
// solo brazos) vs dinamico (balance real por COM + orientacion), giro de
// cintura por yaw del torso, mismas teclas T/M/K/B/R/ESC.
//
// Uso:
//   Terminal 1: python3 pose_sender.py
//   Terminal 2: ./teleop_r1 <ruta_r1.xml>
// ---------------------------------------------------------------------------

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>

#include <array>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <mutex>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

// ===========================================================================
// CONSTANTES  (copiadas 1:1 de teleop_r1_v5.py)
// ===========================================================================
static constexpr int NUM_DOFS = 24;
static constexpr double DT = 0.002;
static constexpr double ARM_UPDATE_HZ = 30.0;

static bool  KINEMATIC_HOLD = true;   // toggle con tecla K
static bool  HOLD_BASE = false;       // toggle con tecla B (solo modo dinamico)

static bool  TRACK_YAW = true;
static constexpr int    WAIST_YAW_INDEX = 13; // VERIFICAR contra r1.xml (13 si se inclina en vez de girar)
static constexpr double YAW_SIGN = 1.0;
static constexpr double YAW_GAIN = 1.0;
static constexpr double YAW_DEADZONE = 0.10;
static constexpr double YAW_CLIP = 0.6;
static constexpr double YAW_EMA = 0.15;

static constexpr double ARM_CONF_LO = 0.75;
static constexpr double ARM_CONF_HI = 1.0;

static constexpr double ARM_EMA = 0.18;
static constexpr double ARM_MAX_STEP = 0.04;

static constexpr int    IK_ITERS = 10;
static constexpr double IK_DAMP  = 3e-3;
static constexpr double IK_EPS   = 1e-4;
static constexpr double IK_GOOD  = 0.12;
static constexpr double IK_BAD   = 0.55;
static constexpr double W_UPPER  = 2.0;

static constexpr double KP_COM_X = 6.0, KD_COM_X = 1.2;
static constexpr double KP_COM_Y = 6.0, KD_COM_Y = 1.2;
static constexpr double COM_CLIP = 0.30;

static constexpr double KP_ROLL = 2.5, KD_ROLL = 0.15;
static constexpr double KP_PITCH = 2.5, KD_PITCH = 0.12;
static constexpr double HIP_ROLL_GAIN = 0.45;
static constexpr double HIP_PITCH_GAIN = 0.25;

static constexpr int L_HIP_PITCH = 0, L_HIP_ROLL = 1;
static constexpr int R_HIP_PITCH = 6, R_HIP_ROLL = 7;
static constexpr int L_ANK_PITCH = 4, L_ANK_ROLL = 5;
static constexpr int R_ANK_PITCH = 10, R_ANK_ROLL = 11;

static const std::array<float, NUM_DOFS> STIFFNESS = {
    120,120,100,220,120,80,  120,120,100,220,120,80,
    250,250,  50,50,40,30,20,  50,50,40,30,20
};
static const std::array<float, NUM_DOFS> DAMPING = {
    6,6,5,9,8,6,  6,6,5,9,8,6,
    25,25,  5,5,4,4,4,  5,5,4,4,4
};
static const std::array<float, NUM_DOFS> TORQUE_LIMITS = {
    88,139,88,139,50,50,  88,139,88,139,50,50,
    88,50,  25,25,25,25,25,  25,25,25,25,25
};
static const std::array<float, NUM_DOFS> IDLE = {
    -0.10f,0.0f,0.0f,0.20f,-0.10f,0.0f,  -0.10f,0.0f,0.0f,0.20f,-0.10f,0.0f,
     0.0f,0.0f,  0.18f,0.18f,0.0f,1.5f,0.0f,  0.18f,-0.18f,0.0f,1.5f,0.0f
};

static const char* ARM_JOINTS_L[4] = {
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint"
};
static const char* ARM_JOINTS_R[4] = {
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint"
};
static const char* ARM_BODIES_L[3] = {
    "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link"
};
static const char* ARM_BODIES_R[3] = {
    "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link"
};
static const char* FOOT_BODIES[2] = { "left_ankle_roll_link", "right_ankle_roll_link" };

static const Eigen::Vector4d Q_IDLE_L(0.18, 0.18, 0.0, 1.5);
static const Eigen::Vector4d Q_IDLE_R(0.18, -0.18, 0.0, 1.5);

// ===========================================================================
// UTILIDADES
// ===========================================================================
static Eigen::Vector3d normv(const Eigen::Vector3d& v) {
    double n = v.norm();
    return n > 1e-9 ? (v / n) : v;
}

static double smoothstep(double x, double lo, double hi) {
    if (x <= lo) return 0.0;
    if (x >= hi) return 1.0;
    double t = (x - lo) / (hi - lo);
    return t * t * (3 - 2 * t);
}

static void quatToEuler(const double q[4], double& roll, double& pitch, double& yaw) {
    double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    roll = std::atan2(2 * (qw*qx + qy*qz), 1 - 2 * (qx*qx + qy*qy));
    double sinp = 2 * (qw*qy - qz*qx);
    pitch = (std::abs(sinp) >= 1) ? std::copysign(M_PI/2, sinp) : std::asin(sinp);
    yaw = std::atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz));
}

static Eigen::Vector3d mirrorDir(const Eigen::Vector3d& v) {
    return Eigen::Vector3d(v[0], -v[1], v[2]);
}

// ===========================================================================
// PAQUETE UDP  (debe coincidir EXACTO con PACKET_FMT="<dI15f" de pose_sender.py)
// ===========================================================================
#pragma pack(push, 1)
struct PoseMsg {
    double   timestamp;
    uint32_t valid;
    float    payload[15]; // yaw, confL, confR, uL(3), wL(3), uR(3), wR(3)
};
#pragma pack(pop)
static_assert(sizeof(PoseMsg) == 72, "PoseMsg debe pesar 72 bytes, igual que pose_sender.py");

class UdpPoseReceiver {
public:
    bool init(int port) {
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) { perror("socket"); return false; }
        int flags = fcntl(sock_, F_GETFL, 0);
        fcntl(sock_, F_SETFL, flags | O_NONBLOCK);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        addr.sin_port = htons(port);
        if (bind(sock_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind"); return false;
        }
        printf("[UDP] Escuchando pose en 127.0.0.1:%d\n", port);
        return true;
    }

    // Drena el socket y se queda con el paquete mas reciente.
    void poll() {
        PoseMsg msg;
        bool got_any = false;
        while (true) {
            ssize_t n = recv(sock_, &msg, sizeof(msg), 0);
            if (n == (ssize_t)sizeof(msg)) {
                got_any = true;
                latest_ = msg;
            } else {
                break; // EAGAIN/EWOULDBLOCK u otro error: no hay mas datos
            }
        }
        if (got_any) last_recv_time_ = std::chrono::steady_clock::now();
    }

    // true si tenemos pose valida y no esta obsoleta (fuente viva).
    bool havePose() const {
        auto age = std::chrono::steady_clock::now() - last_recv_time_;
        bool fresh = age < std::chrono::milliseconds(300);
        return fresh && latest_.valid == 1;
    }

    float yaw() const { return latest_.payload[0]; }
    float confL() const { return latest_.payload[1]; }
    float confR() const { return latest_.payload[2]; }
    Eigen::Vector3d uL() const { return {latest_.payload[3], latest_.payload[4], latest_.payload[5]}; }
    Eigen::Vector3d wL() const { return {latest_.payload[6], latest_.payload[7], latest_.payload[8]}; }
    Eigen::Vector3d uR() const { return {latest_.payload[9], latest_.payload[10], latest_.payload[11]}; }
    Eigen::Vector3d wR() const { return {latest_.payload[12], latest_.payload[13], latest_.payload[14]}; }

private:
    int sock_ = -1;
    PoseMsg latest_{};
    std::chrono::steady_clock::time_point last_recv_time_{};
};

// ===========================================================================
// ArmIK  -- puerto directo de la clase ArmIK de Python (Gauss-Newton numerico)
// ===========================================================================
class ArmIK {
public:
    void load(const std::string& xml_path) {
        char error[1000] = "";
        model_ = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
        if (!model_) { fprintf(stderr, "ArmIK: error cargando XML: %s\n", error); std::exit(1); }
        data_ = mj_makeData(model_);

        setupSide(0, ARM_JOINTS_L, ARM_BODIES_L);
        setupSide(1, ARM_JOINTS_R, ARM_BODIES_R);
    }

    // side: 0=left, 1=right
    std::pair<Eigen::Vector4d,double> solve(int side, const Eigen::Vector3d& u_t,
                                             const Eigen::Vector3d& w_t,
                                             const Eigen::Vector4d& q_prev) {
        auto [q, err] = gaussNewton(side, u_t, w_t, q_prev);
        if (err < IK_GOOD) return {q, err};

        double s = (side == 0) ? 1.0 : -1.0;
        Eigen::Vector4d starts[4] = {
            {0.0, 0.05*s, 0.0, 0.15},
            {-1.2, 0.10*s, 0.0, 0.60},
            {0.0, 1.20*s, 0.0, 0.40},
            (side == 0 ? Q_IDLE_L : Q_IDLE_R)
        };
        for (auto& s0 : starts) {
            auto [qi, ei] = gaussNewton(side, u_t, w_t, s0);
            if (ei < err) { q = qi; err = ei; }
            if (err < IK_GOOD) break;
        }
        return {q, err};
    }

private:
    void setupSide(int side, const char* joints[4], const char* bodies[3]) {
        for (int i = 0; i < 4; ++i) {
            int jid = mj_name2id(model_, mjOBJ_JOINT, joints[i]);
            qadr_[side][i] = model_->jnt_qposadr[jid];
            lo_[side][i] = model_->jnt_range[jid*2 + 0];
            hi_[side][i] = model_->jnt_range[jid*2 + 1];
        }
        for (int i = 0; i < 3; ++i)
            bodyid_[side][i] = mj_name2id(model_, mjOBJ_BODY, bodies[i]);
    }

    // Devuelve (u,w) = direcciones normalizadas hombro->codo, codo->muneca
    void fk(int side, const Eigen::Vector4d& q, Eigen::Vector3d& u, Eigen::Vector3d& w) {
        mju_zero(data_->qpos, model_->nq);
        data_->qpos[3] = 1.0; // quat identidad
        for (int i = 0; i < 4; ++i) data_->qpos[qadr_[side][i]] = q[i];
        mj_forward(model_, data_);

        auto xpos = [&](int b) { return Eigen::Vector3d(data_->xpos[3*b], data_->xpos[3*b+1], data_->xpos[3*b+2]); };
        Eigen::Vector3d sh = xpos(bodyid_[side][0]);
        Eigen::Vector3d el = xpos(bodyid_[side][1]);
        Eigen::Vector3d wr = xpos(bodyid_[side][2]);
        u = normv(el - sh);
        w = normv(wr - el);
    }

    Eigen::Matrix<double,6,1> residual(int side, const Eigen::Vector4d& q,
                                        const Eigen::Vector3d& u_t, const Eigen::Vector3d& w_t) {
        Eigen::Vector3d u, w;
        fk(side, q, u, w);
        Eigen::Matrix<double,6,1> r;
        r.head<3>() = W_UPPER * (u - u_t);
        r.tail<3>() = w - w_t;
        return r;
    }

    std::pair<Eigen::Vector4d,double> gaussNewton(int side, const Eigen::Vector3d& u_t,
                                                   const Eigen::Vector3d& w_t,
                                                   Eigen::Vector4d q0) {
        Eigen::Vector4d q;
        for (int i = 0; i < 4; ++i) q[i] = std::clamp(q0[i], lo_[side][i], hi_[side][i]);

        Eigen::Matrix<double,6,1> r = residual(side, q, u_t, w_t);
        for (int it = 0; it < IK_ITERS; ++it) {
            if (r.norm() < 1e-3) break;
            Eigen::Matrix<double,6,4> J;
            for (int i = 0; i < 4; ++i) {
                Eigen::Vector4d dq = q; dq[i] += IK_EPS;
                J.col(i) = (residual(side, dq, u_t, w_t) - r) / IK_EPS;
            }
            Eigen::Matrix4d H = J.transpose() * J + IK_DAMP * Eigen::Matrix4d::Identity();
            Eigen::Vector4d step = H.ldlt().solve(-(J.transpose() * r));
            for (int i = 0; i < 4; ++i) q[i] = std::clamp(q[i] + step[i], lo_[side][i], hi_[side][i]);
            r = residual(side, q, u_t, w_t);
        }
        return {q, r.norm()};
    }

    mjModel* model_ = nullptr;
    mjData*  data_  = nullptr;
    int    qadr_[2][4]{};
    double lo_[2][4]{}, hi_[2][4]{};
    int    bodyid_[2][3]{};
};

// ===========================================================================
// ESTADO GLOBAL DEL VISOR  (identico patron a play_r1_isaac.cpp)
// ===========================================================================
static mjModel* m = nullptr;
static mjData*  d = nullptr;
static mjvCamera cam;
static mjvOption opt;
static mjvScene  scn;
static mjrContext con;
static bool button_left = false, button_middle = false, button_right = false;
static double lastx = 0, lasty = 0;

// ===========================================================================
// R1Teleop  -- puerto directo de la clase R1Teleop de Python
// ===========================================================================
class R1Teleop {
public:
    void init(const std::string& xml_path) {
        char error[1000] = "";
        printf("Cargando modelo: %s\n", xml_path.c_str());
        m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
        if (!m) { fprintf(stderr, "ERROR: %s\n", error); std::exit(1); }
        m->opt.timestep = DT;
        d = mj_makeData(m);

        for (int i = 0; i < 2; ++i) foot_id_[i] = mj_name2id(m, mjOBJ_BODY, FOOT_BODIES[i]);

        printf("Preparando IK...\n");
        ik_.load(xml_path);

        int gyro_id = mj_name2id(m, mjOBJ_SENSOR, "imu_ang_vel");
        has_gyro_ = gyro_id >= 0;
        if (has_gyro_) gyro_adr_ = m->sensor_adr[gyro_id];

        udp_.init(5555);
        reset();
    }

    void reset() {
        mj_resetData(m, d);
        d->qpos[0] = 0.0; d->qpos[1] = 0.0; d->qpos[2] = 0.74;
        d->qpos[3] = 1.0; d->qpos[4] = 0.0; d->qpos[5] = 0.0; d->qpos[6] = 0.0;
        for (int i = 0; i < NUM_DOFS; ++i) d->qpos[7+i] = IDLE[i];
        for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
        mj_forward(m, d);

        arm_target_.fill(0.0f);
        for (int i = 0; i < 10; ++i) arm_target_[i] = IDLE[14+i];
        waist_yaw_target_ = 0.0;
        q_ik_L_ = Q_IDLE_L; q_ik_R_ = Q_IDLE_R;
        com_err_prev_.setZero();
        for (int i = 0; i < 7; ++i) base0_[i] = d->qpos[i];
    }

    // --- teclado (llamado desde el callback GLFW) ---
    void onKey(int key) {
        switch (key) {
            case GLFW_KEY_T:
                teleop_on_ = !teleop_on_;
                printf("\n[TELEOP] %s\n", teleop_on_ ? "ON" : "OFF");
                break;
            case GLFW_KEY_M:
                mirror_ = !mirror_;
                printf("\n[MIRROR] %s\n", mirror_ ? "ON" : "OFF");
                break;
            case GLFW_KEY_K:
                KINEMATIC_HOLD = !KINEMATIC_HOLD;
                reset();
                printf("\n[MODO] %s\n", KINEMATIC_HOLD ? "CINEMATICO (fijo, solo brazos)" : "DINAMICO (equilibrio real)");
                break;
            case GLFW_KEY_B:
                HOLD_BASE = !HOLD_BASE;
                printf("\n[SOPORTE] %s (solo aplica en modo dinamico)\n", HOLD_BASE ? "ON" : "OFF");
                break;
            case GLFW_KEY_R:
                reset();
                printf("\n[RESET]\n");
                break;
        }
    }

    void step() {
        udp_.poll();
        if (step_count_ % decim_ == 0) {
            updateWaistYaw();
            updateArms();
        }
        if (KINEMATIC_HOLD) stepKinematic();
        else stepDynamic();
        ++step_count_;
    }

private:
    void updateWaistYaw() {
        if (!TRACK_YAW) { waist_yaw_target_ = 0.0; return; }
        double yaw = (teleop_on_ && udp_.havePose()) ? (double)udp_.yaw() : 0.0;
        if (mirror_) yaw = -yaw;
        if (std::abs(yaw) < YAW_DEADZONE) yaw = 0.0;
        else yaw -= std::copysign(YAW_DEADZONE, yaw);
        double tgt = std::clamp(YAW_SIGN * YAW_GAIN * yaw, -YAW_CLIP, YAW_CLIP);
        waist_yaw_target_ = (1 - YAW_EMA) * waist_yaw_target_ + YAW_EMA * tgt;
    }

    void updateArms() {
        std::array<double, 10> desired{};
        bool have = teleop_on_ && udp_.havePose();

        if (!have) {
            q_ik_L_ = Q_IDLE_L; q_ik_R_ = Q_IDLE_R;
            for (int i = 0; i < 4; ++i) desired[i]   = IDLE[14+i];
            for (int i = 0; i < 4; ++i) desired[5+i] = IDLE[19+i];
        } else {
            Eigen::Vector3d uL = udp_.uL(), wL = udp_.wL(), uR = udp_.uR(), wR = udp_.wR();
            double gL = smoothstep(udp_.confL(), ARM_CONF_LO, ARM_CONF_HI);
            double gR = smoothstep(udp_.confR(), ARM_CONF_LO, ARM_CONF_HI);

            Eigen::Vector3d tL_u, tL_w, tR_u, tR_w;
            if (mirror_) {
                tL_u = mirrorDir(uR); tL_w = mirrorDir(wR);
                tR_u = mirrorDir(uL); tR_w = mirrorDir(wL);
                std::swap(gL, gR);
            } else {
                tL_u = uL; tL_w = wL; tR_u = uR; tR_w = wR;
            }

            auto [qL, eL] = ik_.solve(0, tL_u, tL_w, q_ik_L_);
            auto [qR, eR] = ik_.solve(1, tR_u, tR_w, q_ik_R_);

            auto now = std::chrono::steady_clock::now();
            if (now - last_print_ > std::chrono::seconds(1)) {
                last_print_ = now;
                printf("\r[IK] izq=%.3f der=%.3f  gate L=%.2f R=%.2f  yaw=%+.2f  [%s]",
                       eL, eR, gL, gR, waist_yaw_target_, KINEMATIC_HOLD ? "CINE" : "DYN ");
                fflush(stdout);
            }

            if (eL < IK_BAD) q_ik_L_ = gL * qL + (1 - gL) * q_ik_L_;
            if (eR < IK_BAD) q_ik_R_ = gR * qR + (1 - gR) * q_ik_R_;

            for (int i = 0; i < 4; ++i) desired[i]   = q_ik_L_[i];
            for (int i = 0; i < 4; ++i) desired[5+i] = q_ik_R_[i];
        }

        std::array<float,10> smoothed{};
        for (int i = 0; i < 10; ++i) {
            double s = (1 - ARM_EMA) * arm_target_[i] + ARM_EMA * desired[i];
            double delta = std::clamp(s - (double)arm_target_[i], -ARM_MAX_STEP, ARM_MAX_STEP);
            arm_target_[i] = (float)(arm_target_[i] + delta);
        }
    }

    // --- balance dinamico: orientacion + centro de masa ---
    std::array<float, NUM_DOFS> balanceTargets() {
        std::array<float, NUM_DOFS> pd = IDLE;
        double roll, pitch, yaw;
        quatToEuler(&d->qpos[3], roll, pitch, yaw);

        double w[3];
        if (has_gyro_) { w[0]=d->sensordata[gyro_adr_]; w[1]=d->sensordata[gyro_adr_+1]; w[2]=d->sensordata[gyro_adr_+2]; }
        else           { w[0]=d->qvel[3]; w[1]=d->qvel[4]; w[2]=d->qvel[5]; }

        Eigen::Vector2d com(d->subtree_com[0], d->subtree_com[1]);
        Eigen::Vector2d f0(d->xpos[3*foot_id_[0]], d->xpos[3*foot_id_[0]+1]);
        Eigen::Vector2d f1(d->xpos[3*foot_id_[1]], d->xpos[3*foot_id_[1]+1]);
        Eigen::Vector2d feet = 0.5 * (f0 + f1);
        Eigen::Vector2d diff = com - feet;

        double c = std::cos(-yaw), s = std::sin(-yaw);
        Eigen::Vector2d err(c*diff[0] - s*diff[1], s*diff[0] + c*diff[1]);
        Eigen::Vector2d derr = (err - com_err_prev_) / DT;
        com_err_prev_ = err;

        double com_pitch = std::clamp(KP_COM_X*err[0] + KD_COM_X*derr[0], -COM_CLIP, COM_CLIP);
        double com_roll  = std::clamp(-(KP_COM_Y*err[1] + KD_COM_Y*derr[1]), -COM_CLIP, COM_CLIP);

        double pitch_eff = pitch + com_pitch;
        double roll_eff  = roll + com_roll;

        double rc = -(KP_ROLL*roll_eff + KD_ROLL*w[0]);
        double pc = -(KP_PITCH*pitch_eff + KD_PITCH*w[1]);

        pd[L_ANK_PITCH] -= (float)pc; pd[R_ANK_PITCH] -= (float)pc;
        pd[L_ANK_ROLL]  += (float)rc; pd[R_ANK_ROLL]  += (float)rc;
        pd[L_HIP_PITCH] -= (float)(pc*HIP_PITCH_GAIN); pd[R_HIP_PITCH] -= (float)(pc*HIP_PITCH_GAIN);
        pd[L_HIP_ROLL]  += (float)(rc*HIP_ROLL_GAIN);  pd[R_HIP_ROLL]  += (float)(rc*HIP_ROLL_GAIN);
        return pd;
    }

    void applyTorque(const std::array<float,NUM_DOFS>& target) {
        for (int i = 0; i < NUM_DOFS; ++i) {
            float q = (float)d->qpos[7+i], qd = (float)d->qvel[6+i];
            float tau = STIFFNESS[i]*(target[i]-q) - DAMPING[i]*qd;
            d->ctrl[i] = std::clamp(tau, -TORQUE_LIMITS[i], TORQUE_LIMITS[i]);
        }
    }

    void stepKinematic() {
        for (int i = 0; i < 7; ++i) d->qpos[i] = base0_[i];
        for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
        for (int i = 0; i < NUM_DOFS; ++i) d->qpos[7+i] = IDLE[i];
        if (TRACK_YAW) d->qpos[7+WAIST_YAW_INDEX] = waist_yaw_target_;
        for (int i = 0; i < 10; ++i) d->qpos[7+14+i] = arm_target_[i];
        mj_forward(m, d);
    }

    void stepDynamic() {
        auto target = balanceTargets();
        if (TRACK_YAW) target[WAIST_YAW_INDEX] = (float)waist_yaw_target_;
        for (int i = 0; i < 10; ++i) target[14+i] = arm_target_[i];
        applyTorque(target);
        mj_step(m, d);
        if (HOLD_BASE) {
            for (int i = 0; i < 7; ++i) d->qpos[i] = base0_[i];
            for (int i = 0; i < 6; ++i) d->qvel[i] = 0.0;
            mj_forward(m, d);
        }
    }

    ArmIK ik_;
    UdpPoseReceiver udp_;
    int foot_id_[2]{};
    bool has_gyro_ = false;
    int gyro_adr_ = -1;

    bool teleop_on_ = true;
    bool mirror_ = true;

    std::array<float,10> arm_target_{};
    double waist_yaw_target_ = 0.0;
    Eigen::Vector4d q_ik_L_ = Q_IDLE_L, q_ik_R_ = Q_IDLE_R;
    Eigen::Vector2d com_err_prev_{0,0};
    double base0_[7]{};

    long step_count_ = 0;
    int decim_ = (int)std::lround((1.0 / ARM_UPDATE_HZ) / DT);
    std::chrono::steady_clock::time_point last_print_{};
};

static R1Teleop g_teleop;

// ===========================================================================
// CALLBACKS GLFW  (identicos a play_r1_isaac.cpp + delega teclas a R1Teleop)
// ===========================================================================
static void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) { glfwSetWindowShouldClose(window, GLFW_TRUE); return; }
    g_teleop.onKey(key);
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
    if (button_right)      action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)  action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else                    action = mjMOUSE_ZOOM;
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

static void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char** argv) {
    std::string xml_path = (argc > 1) ? argv[1] : "assets/r1.xml";

    g_teleop.init(xml_path);

    if (!glfwInit()) { fprintf(stderr, "No se pudo iniciar GLFW\n"); return 1; }
    GLFWwindow* window = glfwCreateWindow(1200, 900, "R1 Teleop (C++)", nullptr, nullptr);
    if (!window) { fprintf(stderr, "No se pudo crear la ventana\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    cam.distance = 3.0; cam.elevation = -15; cam.azimuth = 180;

    glfwSetKeyCallback(window, keyboard);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetScrollCallback(window, scroll);

    printf("\n== TELEOP R1 (C++) ==\n");
    printf("T: on/off | M: espejo | K: cine/dinamico | B: soporte | R: reset | ESC: salir\n");
    printf("Corre pose_sender.py en otra terminal para alimentar la pose.\n\n");

    auto t_prev = std::chrono::steady_clock::now();
        double accumulator = 0.0;
        static constexpr double MAX_FRAME_TIME = 0.25; // evita "spiral of death" tras un frame lento

        while (!glfwWindowShouldClose(window)) {
            auto t_now = std::chrono::steady_clock::now();
            double frame_time = std::chrono::duration<double>(t_now - t_prev).count();
            t_prev = t_now;
            if (frame_time > MAX_FRAME_TIME) frame_time = MAX_FRAME_TIME;
            accumulator += frame_time;

            // Varios pasos de fisica (DT=2ms) por cada frame renderizado, para
            // que la simulacion corra a tiempo real independientemente de la
            // tasa de refresco/vsync del monitor.
            while (accumulator >= DT) {
                g_teleop.step();
                accumulator -= DT;
            }

            cam.lookat[0] = d->qpos[0]; cam.lookat[1] = d->qpos[1]; cam.lookat[2] = d->qpos[2];
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            mjrRect viewport = {0, 0, width, height};
            mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        auto t0 = std::chrono::steady_clock::now();

        g_teleop.step();

        cam.lookat[0] = d->qpos[0]; cam.lookat[1] = d->qpos[1]; cam.lookat[2] = d->qpos[2];
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        mjrRect viewport = {0, 0, width, height};
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();

        auto elapsed = std::chrono::steady_clock::now() - t0;
        auto sleep_dur = std::chrono::duration<double>(DT) - elapsed;
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