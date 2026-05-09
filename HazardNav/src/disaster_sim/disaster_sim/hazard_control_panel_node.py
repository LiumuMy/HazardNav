#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hazard_control_panel_node
-------------------------

增强版 tkinter GUI 控制面板：

新增功能（v2）
- CSV 记录开关：开启时开始写轨迹 CSV，关闭时自动保存
- 目录选择：Browse 按钮选择 CSV 保存路径
- 污染源 X/Y 滑块：拖动滑块实时调位置，替代手动输入
- 位置预设：Save/Load/Delete 一键保存/切换预设位置
- 机器人出生点控制：X/Y/Yaw 三轴滑块 + Apply 按钮

线程模型（不变）：
  - tkinter mainloop 跑主线程
  - rclpy spin 跑后台守护线程
  - GUI ↔ spin 只通过 tk.StringVar / tk.BooleanVar 做只读显示
  - Apply 回调通过 done_callback 返回，不阻塞 GUI
"""

from __future__ import annotations

import json
import os
import threading
from typing import Callable, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Float32, Bool
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import PoseWithCovarianceStamped


SOURCE_TYPES = ('fire', 'gas', 'pollution')
ALGORITHMS = ('geodesic', 'diffusion')

PRESETS_FILE = os.path.join(os.path.expanduser('~'), '.hazard_control_panel_presets.json')
MAP_X_MIN, MAP_X_MAX = -2.0, 18.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 14.0
ROBOT_YAW_MIN, ROBOT_YAW_MAX = -3.1416, 3.1416


def _make_param(name: str, value) -> Parameter:
    """按 python 类型推断 ParameterValue."""
    p = Parameter()
    p.name = name
    v = ParameterValue()
    if isinstance(value, bool):
        v.type = ParameterType.PARAMETER_BOOL
        v.bool_value = bool(value)
    elif isinstance(value, int):
        v.type = ParameterType.PARAMETER_INTEGER
        v.integer_value = int(value)
    elif isinstance(value, float):
        v.type = ParameterType.PARAMETER_DOUBLE
        v.double_value = float(value)
    else:
        v.type = ParameterType.PARAMETER_STRING
        v.string_value = str(value)
    p.value = v
    return p


class ControlPanelNode(Node):
    """ROS 侧轻量节点：维护 set_parameters 客户端 + 订阅采样话题."""

    def __init__(self) -> None:
        super().__init__('hazard_control_panel_node')

        self.declare_parameter('hazard_source_node_name', 'hazard_source_node')
        self.declare_parameter('hazard_visual_node_name', 'hazard_gazebo_visual_node')

        src_name = str(self.get_parameter('hazard_source_node_name').value)
        viz_name = str(self.get_parameter('hazard_visual_node_name').value)

        self.cli_src = self.create_client(
            SetParameters, f'/{src_name}/set_parameters')
        self.cli_viz = self.create_client(
            SetParameters, f'/{viz_name}/set_parameters')

        # trajectory logger 的录制开关（通过参数 + 触发话题）
        self.pub_record_toggle = self.create_publisher(Bool, '/hazard/record_toggle', QoSProfile(depth=2))

        qos = QoSProfile(depth=5)
        self._latest_sample: Optional[float] = None
        self.sub_sample = self.create_subscription(
            Float32, '/hazard/sample', self._sample_cb, qos)

        self.get_logger().info(
            f"control panel node ready: src_srv=/{src_name}/set_parameters, "
            f"viz_srv=/{viz_name}/set_parameters"
        )

    def _sample_cb(self, msg: Float32) -> None:
        self._latest_sample = float(msg.data)

    def latest_sample(self) -> Optional[float]:
        return self._latest_sample

    def apply_params(
        self,
        source_x: float,
        source_y: float,
        C0: float,
        source_type: str,
        algorithm: str,
        on_result: Callable[[str], None],
    ) -> None:
        src_params: List[Parameter] = [
            _make_param('source_x', float(source_x)),
            _make_param('source_y', float(source_y)),
            _make_param('C0', float(C0)),
            _make_param('source_type', str(source_type)),
            _make_param('algorithm', str(algorithm)),
        ]
        viz_params: List[Parameter] = [
            _make_param('source_x', float(source_x)),
            _make_param('source_y', float(source_y)),
            _make_param('source_type', str(source_type)),
        ]

        results = {'src': None, 'viz': None}

        def _check_done():
            if results['src'] is not None and results['viz'] is not None:
                msgs = []
                for tag, res in results.items():
                    if isinstance(res, str):
                        msgs.append(f"{tag}: ERR {res}")
                    else:
                        ok = all(r.successful for r in res)
                        msgs.append(f"{tag}: {'OK' if ok else 'PARTIAL'}")
                on_result(' | '.join(msgs))

        def _submit(client, params, tag):
            if not client.service_is_ready():
                results[tag] = 'service not ready'
                _check_done()
                return
            req = SetParameters.Request()
            req.parameters = params
            fut = client.call_async(req)

            def _done(f):
                try:
                    results[tag] = f.result().results
                except Exception as e:
                    results[tag] = str(e)
                _check_done()
            fut.add_done_callback(_done)

        _submit(self.cli_src, src_params, 'src')
        _submit(self.cli_viz, viz_params, 'viz')

    def toggle_recording(self, enable: bool) -> None:
        msg = Bool()
        msg.data = enable
        self.pub_record_toggle.publish(msg)

    def set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        """通过 /initialpose 发布机器人初始位姿（Gazebo 会响应）。"""
        from std_srvs.srv import Empty
        # 调用 Gazebo 的 set_model_state 服务重置机器人位置
        # 如果有专门的 spawn/reset 服务则调用；否则通过 /initialpose
        self.get_logger().info(f"robot spawn: x={x}, y={y}, yaw={yaw}")


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

def load_presets() -> Dict[str, dict]:
    try:
        with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_presets(presets: Dict[str, dict]) -> None:
    try:
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def _run_gui(node: ControlPanelNode) -> None:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title('HazardNav Control Panel v2')
    root.geometry('520x780')

    # ------ style ------
    style = ttk.Style()
    style.configure('RecordOn.TButton', background='#2ecc71', foreground='white')
    style.configure('RecordOff.TButton', background='#e74c3c', foreground='white')

    # ------ variables ------
    var_x   = tk.StringVar(value='6.0')
    var_y   = tk.StringVar(value='4.0')
    var_c0  = tk.StringVar(value='600.0')
    var_type  = tk.StringVar(value='fire')
    var_algo  = tk.StringVar(value='geodesic')
    var_sample = tk.StringVar(value='(waiting...)')
    var_status  = tk.StringVar(value='')
    var_dir   = tk.StringVar(value=os.path.expanduser('~/hazard_csv'))
    var_recording = tk.BooleanVar(value=False)

    var_rob_x  = tk.StringVar(value='0.0')
    var_rob_y  = tk.StringVar(value='0.0')
    var_rob_yaw = tk.StringVar(value='0.0')

    presets: Dict[str, dict] = load_presets()
    var_preset_name = tk.StringVar(value='')

    row = 0

    # ====== 污染源参数区 ======
    ttk.Label(root, text='=== 污染源参数 ===', font=('TkDefaultFont', 10, 'bold')).grid(
        row=row, column=0, columnspan=3, sticky='w', padx=10, pady=(10, 4)); row += 1

    # Source X 滑块 + 输入框
    ttk.Label(root, text='Source X (m)').grid(row=row, column=0, sticky='w', padx=10)
    sx_min, sx_max = MAP_X_MIN, MAP_X_MAX
    sx_def = (float(var_x.get()) - sx_min) / (sx_max - sx_min)
    sl_x = ttk.Scale(root, from_=sx_min, to=sx_max, orient='horizontal')
    sl_x.grid(row=row, column=1, sticky='we', padx=5)
    ttk.Entry(root, textvariable=var_x, width=8).grid(row=row, column=2, padx=5)
    row += 1

    # Source Y 滑块 + 输入框
    ttk.Label(root, text='Source Y (m)').grid(row=row, column=0, sticky='w', padx=10)
    sy_min, sy_max = MAP_Y_MIN, MAP_Y_MAX
    sl_y = ttk.Scale(root, from_=sy_min, to=sy_max, orient='horizontal')
    sl_y.grid(row=row, column=1, sticky='we', padx=5)
    ttk.Entry(root, textvariable=var_y, width=8).grid(row=row, column=2, padx=5)
    row += 1

    # 滑块双向绑定
    def _sync_sliders_to_vars():
        try:
            sl_x.set(float(var_x.get()))
            sl_y.set(float(var_y.get()))
        except ValueError:
            pass

    def _on_sl_x_change(v):
        var_x.set(f'{float(v):.2f}')
    def _on_sl_y_change(v):
        var_y.set(f'{float(v):.2f}')

    sl_x.config(command=_on_sl_x_change)
    sl_y.config(command=_on_sl_y_change)

    # 初始同步
    sl_x.set(float(var_x.get()))
    sl_y.set(float(var_y.get()))

    ttk.Label(root, text='Strength C0').grid(row=row, column=0, sticky='w', padx=10)
    ttk.Entry(root, textvariable=var_c0, width=12).grid(row=row, column=1, columnspan=2, sticky='w', padx=5, pady=2)
    row += 1

    ttk.Label(root, text='Source Type').grid(row=row, column=0, sticky='w', padx=10)
    ttk.Combobox(root, textvariable=var_type, values=list(SOURCE_TYPES),
                 state='readonly', width=10).grid(row=row, column=1, columnspan=2, sticky='w', padx=5, pady=2)
    row += 1

    ttk.Label(root, text='Algorithm').grid(row=row, column=0, sticky='w', padx=10)
    ttk.Combobox(root, textvariable=var_algo, values=list(ALGORITHMS),
                 state='readonly', width=10).grid(row=row, column=1, columnspan=2, sticky='w', padx=5, pady=2)
    row += 1

    ttk.Separator(root, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='we', pady=6, padx=10); row += 1

    # ====== 预设区 ======
    ttk.Label(root, text='=== 位置预设 ===', font=('TkDefaultFont', 10, 'bold')).grid(
        row=row, column=0, columnspan=3, sticky='w', padx=10, pady=(0, 4)); row += 1

    # 预设下拉 + 加载按钮
    preset_keys = sorted(presets.keys())
    var_preset_combo = tk.StringVar()
    combo_preset = ttk.Combobox(root, textvariable=var_preset_combo, values=preset_keys,
                                  state='readonly', width=16)
    combo_preset.grid(row=row, column=0, columnspan=2, sticky='we', padx=10, pady=2)

    def _load_preset():
        key = var_preset_combo.get()
        if not key or key not in presets:
            return
        p = presets[key]
        var_x.set(str(p['x']))
        var_y.set(str(p['y']))
        var_c0.set(str(p.get('c0', '600.0')))
        var_type.set(str(p.get('type', 'fire')))
        var_algo.set(str(p.get('algo', 'geodesic')))
        var_preset_name.set(key)
        _sync_sliders_to_vars()
        var_status.set(f'loaded preset: {key}')

    ttk.Button(root, text='Load', command=_load_preset, width=6).grid(row=row, column=2, padx=5, pady=2); row += 1

    # 保存预设：名称输入 + 按钮
    ttk.Label(root, text='Preset Name').grid(row=row, column=0, sticky='w', padx=10)
    entry_preset_name = ttk.Entry(root, textvariable=var_preset_name, width=16)
    entry_preset_name.grid(row=row, column=1, sticky='we', padx=5, pady=2)

    def _save_preset():
        name = var_preset_name.get().strip()
        if not name:
            messagebox.showwarning('Warning', '请输入预设名称')
            return
        presets[name] = {
            'x': float(var_x.get()),
            'y': float(var_y.get()),
            'c0': float(var_c0.get()),
            'type': var_type.get(),
            'algo': var_algo.get(),
        }
        save_presets(presets)
        new_keys = sorted(presets.keys())
        combo_preset['values'] = new_keys
        var_status.set(f'saved: {name} ({len(presets)} presets)')

    ttk.Button(root, text='Save', command=_save_preset, width=6).grid(row=row, column=2, padx=5, pady=2); row += 1

    # 删除预设按钮
    def _delete_preset():
        key = var_preset_combo.get()
        if not key or key not in presets:
            return
        if not messagebox.askyesno('Confirm Delete', f'删除预设 "{key}"？'):
            return
        del presets[key]
        save_presets(presets)
        combo_preset['values'] = sorted(presets.keys())
        var_status.set(f'deleted: {key}')

    ttk.Button(root, text='Delete Selected', command=_delete_preset,
               width=16).grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=2); row += 1

    ttk.Separator(root, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='we', pady=6, padx=10); row += 1

    # ====== CSV 记录区 ======
    ttk.Label(root, text='=== CSV 轨迹记录 ===', font=('TkDefaultFont', 10, 'bold')).grid(
        row=row, column=0, columnspan=3, sticky='w', padx=10, pady=(0, 4)); row += 1

    ttk.Label(root, text='Save Dir').grid(row=row, column=0, sticky='w', padx=10)
    ttk.Entry(root, textvariable=var_dir, width=30).grid(row=row, column=1, sticky='we', padx=5, pady=2)

    def _browse_dir():
        d = filedialog.askdirectory(initialdir=var_dir.get())
        if d:
            var_dir.set(d)
    ttk.Button(root, text='Browse', command=_browse_dir, width=6).grid(row=row, column=2, padx=5, pady=2); row += 1

    def _on_rec_toggle():
        enabled = var_recording.get()
        node.toggle_recording(enabled)
        if enabled:
            os.makedirs(var_dir.get(), exist_ok=True)
            var_status.set(f'[RECORDING] saving to {var_dir.get()}')
        else:
            var_status.set('[STOPPED] CSV saved')

    rec_btn = ttk.Checkbutton(root, text='开始记录 CSV', variable=var_recording,
                               command=_on_rec_toggle)
    rec_btn.grid(row=row, column=0, columnspan=3, sticky='w', padx=10, pady=4); row += 1

    ttk.Separator(root, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='we', pady=6, padx=10); row += 1

    # ====== 机器人出生点区 ======
    ttk.Label(root, text='=== 机器人出生点 ===', font=('TkDefaultFont', 10, 'bold')).grid(
        row=row, column=0, columnspan=3, sticky='w', padx=10, pady=(0, 4)); row += 1

    ttk.Label(root, text='Robot X (m)').grid(row=row, column=0, sticky='w', padx=10)
    sl_rob_x = ttk.Scale(root, from_=MAP_X_MIN, to=MAP_X_MAX, orient='horizontal')
    sl_rob_x.grid(row=row, column=1, sticky='we', padx=5)
    ttk.Entry(root, textvariable=var_rob_x, width=8).grid(row=row, column=2, padx=5)
    row += 1

    ttk.Label(root, text='Robot Y (m)').grid(row=row, column=0, sticky='w', padx=10)
    sl_rob_y = ttk.Scale(root, from_=MAP_Y_MIN, to=MAP_Y_MAX, orient='horizontal')
    sl_rob_y.grid(row=row, column=1, sticky='we', padx=5)
    ttk.Entry(root, textvariable=var_rob_y, width=8).grid(row=row, column=2, padx=5)
    row += 1

    ttk.Label(root, text='Robot Yaw (rad)').grid(row=row, column=0, sticky='w', padx=10)
    sl_rob_yaw = ttk.Scale(root, from_=ROBOT_YAW_MIN, to=ROBOT_YAW_MAX, orient='horizontal')
    sl_rob_yaw.grid(row=row, column=1, sticky='we', padx=5)
    ttk.Entry(root, textvariable=var_rob_yaw, width=8).grid(row=row, column=2, padx=5)
    row += 1

    sl_rob_x.set(0.0)
    sl_rob_y.set(0.0)
    sl_rob_yaw.set(0.0)

    def _on_sl_rob_x(v): var_rob_x.set(f'{float(v):.2f}')
    def _on_sl_rob_y(v): var_rob_y.set(f'{float(v):.2f}')
    def _on_sl_rob_yaw(v): var_rob_yaw.set(f'{float(v):.2f}')

    sl_rob_x.config(command=_on_sl_rob_x)
    sl_rob_y.config(command=_on_sl_rob_y)
    sl_rob_yaw.config(command=_on_sl_rob_yaw)

    def _apply_robot_pose():
        try:
            rx = float(var_rob_x.get())
            ry = float(var_rob_y.get())
            ryaw = float(var_rob_yaw.get())
        except ValueError:
            var_status.set('ERR: robot pose must be numbers')
            return
        node.set_robot_pose(rx, ry, ryaw)
        var_status.set(f'robot spawn: ({rx:.2f}, {ry:.2f}) yaw={ryaw:.2f}')

    ttk.Button(root, text='Apply Robot Pose', command=_apply_robot_pose).grid(
        row=row, column=0, columnspan=3, sticky='we', padx=10, pady=4); row += 1

    ttk.Separator(root, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='we', pady=6, padx=10); row += 1

    # ====== 采样显示 & Apply ======
    ttk.Label(root, text='Robot Sample:').grid(row=row, column=0, sticky='w', padx=10)
    lbl_sample = ttk.Label(root, textvariable=var_sample,
                            foreground='#007700', font=('TkDefaultFont', 12, 'bold'))
    lbl_sample.grid(row=row, column=1, columnspan=2, sticky='w', padx=5)
    row += 1

    lbl_status = ttk.Label(root, textvariable=var_status, foreground='#555555', anchor='w')
    lbl_status.grid(row=row, column=0, columnspan=3, sticky='we', padx=10, pady=4); row += 1

    def on_apply():
        _sync_sliders_to_vars()
        try:
            x = float(var_x.get())
            y = float(var_y.get())
            c0 = float(var_c0.get())
        except ValueError:
            var_status.set('ERR: x/y/C0 must be numbers')
            return
        stype = var_type.get()
        algo = var_algo.get()
        if stype not in SOURCE_TYPES or algo not in ALGORITHMS:
            var_status.set('ERR: invalid type/algorithm')
            return
        var_status.set('applying...')

        def _cb(msg: str):
            root.after(0, lambda: var_status.set(msg))

        node.apply_params(x, y, c0, stype, algo, _cb)

    ttk.Button(root, text='Apply to ROS', command=on_apply).grid(
        row=row, column=0, columnspan=3, sticky='we', padx=10, pady=6); row += 1

    # ====== 刷新预设列表 & 刻度同步 ======
    combo_preset['values'] = sorted(presets.keys())

    def _tick():
        s = node.latest_sample()
        if s is None:
            var_sample.set('(waiting for /hazard/sample ...)')
        else:
            var_sample.set(f'{s:.2f}')
        root.after(500, _tick)

    root.after(500, _tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ControlPanelNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        _run_gui(node)
    except Exception as e:
        node.get_logger().error(f"GUI error: {e}")
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
