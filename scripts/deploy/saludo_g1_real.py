#!/usr/bin/env python3
"""
@file g1_wave_stable.py
@brief G1 - Saludo con brazo derecho (con balance)
"""

import time
import math
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


G1_NUM_MOTOR = 29


class State:
    INIT = 0
    RAISE_ARM = 1
    WAVING = 2
    LOWER_ARM = 3
    DONE = 4


class G1WaveStable:
    
    def __init__(self, interface, id):
        ChannelFactoryInitialize(id, interface)
        
        self.control_dt = 0.002
        
        # Ganancias
        self.kp = 500.0
        self.kd = 5.0
        
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.ready = False
        
        # Estado
        self.state = State.INIT
        self.state_time = 0.0
        self.wave_count = 0
        
        # IMU
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_roll = 0.0
        self.pitch_vel = 0.0
        self.roll_vel = 0.0
        
        # Posiciones
        self.motor_pos = [0.0] * G1_NUM_MOTOR
        self.cmd_pos = [0.0] * G1_NUM_MOTOR
        self.target_pos = [0.0] * G1_NUM_MOTOR
        self.init_pos = [0.0] * G1_NUM_MOTOR
        
        # Interpolación suave
        self.smooth_factor = 0.03
        
        # ===== BALANCE =====
        self.roll_gain = 0.0
        self.pitch_gain = 0.0
        self.roll_d_gain = 0.0
        self.pitch_d_gain = 0.05
        
        # ===== POSE BASE =====
        self.base_pose = {
            G1JointIndex.LeftHipPitch: -0.35,
            G1JointIndex.LeftHipRoll: 0.12,
            G1JointIndex.LeftHipYaw: 0.0,
            G1JointIndex.LeftKnee: 0.5,
            G1JointIndex.LeftAnklePitch: -0.25,
            G1JointIndex.LeftAnkleRoll: -0.06,
            
            G1JointIndex.RightHipPitch: -0.35,
            G1JointIndex.RightHipRoll: -0.12,
            G1JointIndex.RightHipYaw: 0.0,
            G1JointIndex.RightKnee: 0.5,
            G1JointIndex.RightAnklePitch: -0.25,
            G1JointIndex.RightAnkleRoll: 0.06,
            
            G1JointIndex.WaistYaw: 0.0,
            G1JointIndex.WaistRoll: 0.0,
            G1JointIndex.WaistPitch: 0.0,
            
            G1JointIndex.LeftShoulderPitch: 0.3,
            G1JointIndex.LeftShoulderRoll: 0.3,
            G1JointIndex.LeftShoulderYaw: 0.0,
            G1JointIndex.LeftElbow: 0.5,
            G1JointIndex.LeftWristRoll: 0.0,
            G1JointIndex.LeftWristPitch: 0.0,
            G1JointIndex.LeftWristYaw: 0.0,
            
            G1JointIndex.RightShoulderPitch: 0.3,
            G1JointIndex.RightShoulderRoll: -0.3,
            G1JointIndex.RightShoulderYaw: 0.0,
            G1JointIndex.RightElbow: 0.5,
            G1JointIndex.RightWristRoll: 0.0,
            G1JointIndex.RightWristPitch: 0.0,
            G1JointIndex.RightWristYaw: 0.0,
        }
        
        # ===== POSE DE SALUDO =====
        self.wave_arm = {
            G1JointIndex.RightShoulderPitch: 0.353,
            G1JointIndex.RightShoulderRoll: -1.56,
            G1JointIndex.RightShoulderYaw: -1.26,
            G1JointIndex.RightElbow: -0.403,
        }
        
        # ===== PARÁMETROS =====
        self.wave_speed = 2.0
        self.elbow_min = -0.403
        self.elbow_max = 1.07
        self.num_waves = 5
        
        # ===== TIEMPOS =====
        self.time_to_stand = 5.0
        self.time_raise_arm = 3.0
        self.time_lower_arm = 3.0
        
        self.print_counter = 0

    def Init(self):
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.thread = RecurrentThread(
            interval=self.control_dt,
            target=self.Loop,
            name="wave"
        )
        
        print("Conectando...")
        while not self.ready:
            time.sleep(0.01)
        
        for i in range(G1_NUM_MOTOR):
            self.init_pos[i] = self.motor_pos[i]
            self.cmd_pos[i] = self.motor_pos[i]
            self.target_pos[i] = self.motor_pos[i]
        
        print("[OK] Conectado")
        self.thread.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        
        if hasattr(msg, 'motor_state'):
            for i in range(G1_NUM_MOTOR):
                if i < len(msg.motor_state):
                    self.motor_pos[i] = msg.motor_state[i].q
        
        if hasattr(msg, 'imu_state'):
            imu = msg.imu_state
            if hasattr(imu, 'rpy') and len(imu.rpy) >= 2:
                self.prev_roll = self.imu_roll
                self.prev_pitch = self.imu_pitch
                self.imu_roll = imu.rpy[0]
                self.imu_pitch = imu.rpy[1]
                self.roll_vel = (self.imu_roll - self.prev_roll) / self.control_dt
                self.pitch_vel = (self.imu_pitch - self.prev_pitch) / self.control_dt
        
        self.ready = True

    def get_balance_correction(self):
        """Corrección de balance PD"""
        roll_corr = -self.roll_gain * self.imu_roll - self.roll_d_gain * self.roll_vel
        pitch_corr = -self.pitch_gain * self.imu_pitch - self.pitch_d_gain * self.pitch_vel
        
        roll_corr = max(-0.12, min(0.12, roll_corr))
        pitch_corr = max(-0.08, min(0.08, pitch_corr))
        
        return roll_corr, pitch_corr

    def apply_balance(self, pos):
        """Aplica balance a posiciones (solo piernas)"""
        roll_c, pitch_c = self.get_balance_correction()
        
        result = list(pos)
        
        # Tobillos
        result[G1JointIndex.LeftAnkleRoll] += roll_c * 0.6
        result[G1JointIndex.RightAnkleRoll] += roll_c * 0.6
        result[G1JointIndex.LeftAnklePitch] += pitch_c * 0.4
        result[G1JointIndex.RightAnklePitch] += pitch_c * 0.4
        
        # Cadera
        result[G1JointIndex.LeftHipRoll] -= roll_c * 0.3
        result[G1JointIndex.RightHipRoll] -= roll_c * 0.3
        
        return result

    def smooth(self, t):
        """Interpolación suave"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def get_arm_pose(self, t):
        """Interpola brazo derecho hacia pose de saludo"""
        arm_pos = {}
        for joint, target in self.wave_arm.items():
            start = self.base_pose[joint]
            arm_pos[joint] = start + (target - start) * t
        return arm_pos

    def Loop(self):
        if self.low_state is None:
            return
        
        self.state_time += self.control_dt
        
        # ===== SIEMPRE MANTENER POSE BASE =====
        for joint, value in self.base_pose.items():
            self.target_pos[joint] = value
        
        # ========== MÁQUINA DE ESTADOS ==========
        
        if self.state == State.INIT:
            t = min(1.0, self.state_time / self.time_to_stand)
            t = self.smooth(t)
            
            for joint in self.base_pose.keys():
                start = self.init_pos[joint]
                end = self.base_pose[joint]
                self.target_pos[joint] = start + (end - start) * t
            
            if self.state_time >= self.time_to_stand:
                self.state = State.RAISE_ARM
                self.state_time = 0.0
                print("\n[ESTADO] Levantando brazo...")
        
        elif self.state == State.RAISE_ARM:
            t = min(1.0, self.state_time / self.time_raise_arm)
            t = self.smooth(t)
            
            arm_pos = self.get_arm_pose(t)
            for joint, value in arm_pos.items():
                self.target_pos[joint] = value
            
            if self.state_time >= self.time_raise_arm:
                self.state = State.WAVING
                self.state_time = 0.0
                self.wave_count = 0
                print("\n[ESTADO] ¡Saludando!")
        
        elif self.state == State.WAVING:
            for joint, value in self.wave_arm.items():
                self.target_pos[joint] = value
            
            wave_progress = math.sin(self.state_time * self.wave_speed * math.pi)
            elbow_range = self.elbow_max - self.elbow_min
            elbow_value = self.elbow_min + (wave_progress + 1) * 0.5 * elbow_range
            self.target_pos[G1JointIndex.RightElbow] = elbow_value
            
            cycles = int(self.state_time * self.wave_speed)
            if cycles > self.wave_count:
                self.wave_count = cycles
                print(f"\n[SALUDO] Wave #{self.wave_count}/{self.num_waves}")
            
            if self.wave_count >= self.num_waves:
                self.state = State.LOWER_ARM
                self.state_time = 0.0
                print("\n[ESTADO] Bajando brazo...")
        
        elif self.state == State.LOWER_ARM:
            t = min(1.0, self.state_time / self.time_lower_arm)
            t = self.smooth(t)
            
            for joint, wave_value in self.wave_arm.items():
                stand_value = self.base_pose[joint]
                self.target_pos[joint] = wave_value + (stand_value - wave_value) * t
            
            if self.state_time >= self.time_lower_arm:
                self.state = State.DONE
                self.state_time = 0.0
                print("\n[ESTADO] ¡Completado!")
        
        # ========== APLICAR BALANCE ==========
        balanced = self.apply_balance(self.target_pos)
        
        # ========== INTERPOLACIÓN SUAVE ==========
        for joint in self.base_pose.keys():
            diff = balanced[joint] - self.cmd_pos[joint]
            self.cmd_pos[joint] += self.smooth_factor * diff
        
        # ========== ENVIAR ==========
        self.send_commands()
        
        # ========== PRINT ==========
        self.print_counter += 1
        if self.print_counter >= 250:
            self.print_counter = 0
            self.print_status()

    def send_commands(self):
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = 0
        
        for i in range(G1_NUM_MOTOR):
            cmd = self.low_cmd.motor_cmd[i]
            cmd.mode = 0x01
            cmd.q = self.cmd_pos[i]
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = self.kp
            cmd.kd = self.kd
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def print_status(self):
        names = ["INIT", "RAISE_ARM", "WAVING", "LOWER_ARM", "DONE"]
        r = math.degrees(self.imu_roll)
        p = math.degrees(self.imu_pitch)
        elbow = self.cmd_pos[G1JointIndex.RightElbow]
        
        if self.state == State.WAVING:
            print(f"\r[{names[self.state]:10}] Codo:{elbow:+.2f} | Wave:{self.wave_count}/{self.num_waves} | Roll:{r:+5.1f}° Pitch:{p:+5.1f}°   ", end="", flush=True)
        else:
            print(f"\r[{names[self.state]:10}] t={self.state_time:.1f}s | Roll:{r:+5.1f}° Pitch:{p:+5.1f}°   ", end="", flush=True)

    def freeze(self):
        for i in range(G1_NUM_MOTOR):
            cmd = self.low_cmd.motor_cmd[i]
            cmd.mode = 0x01
            cmd.q = self.motor_pos[i]
            cmd.kp = 0.0
            cmd.kd = 8.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)


def main():
    print("\n" + "="*55)
    print("  G1 - SALUDO")
    print("="*55)
    
    while True:
        try:
            num = input("\n¿Cuántas veces quieres que salude? : ")
            num_waves = int(num)
            if num_waves >= 1:
                break
            else:
                print("Por favor, ingresa un número mayor a 0")
        except ValueError:
            print("Por favor, ingresa un número válido")
    
    print(f"\n[OK] El robot saludará {num_waves} veces")
    
    if len(sys.argv) > 1:
        interface = sys.argv[1]
        id = 0
    else:
        interface = "lo"
        id = 1
        print("[MODO] Simulación MuJoCo")
    
    robot = G1WaveStable(interface, id)
    robot.num_waves = num_waves
    robot.Init()
    
    print("\n" + "="*55)
    print("  Secuencia:")
    print("    1. INIT      → Pararse (5s)")
    print("    2. RAISE_ARM → Levantar brazo (3s)")
    print(f"    3. WAVING    → Saludar ({num_waves} veces)")
    print("    4. LOWER_ARM → Bajar brazo (3s)")
    print("    5. DONE      → Terminado")
    print("="*55 + "\n")
    
    robot.Start()
    
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        robot.freeze()
        print("\n\n[OK] Terminado")


if __name__ == "__main__":
    main()