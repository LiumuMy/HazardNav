#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hazard_control_panel_node
-------------------------

一个 tkinter GUI 控制面板, 让用户实时修改 **污染源的位置 / 强度 / 类型 / 算法**,
并把修改通过 ROS2 `SetParameters` 服务推送给:
  - /hazard_source_node            (负责扩散场计算)
  - /hazard_gazebo_visual_node     (负责 Gazebo 里的火球/气体云/污染球)

效果:
  - 点 Apply → 扩散场立刻按新参数重算, RViz 里热力图即时刷新
  - Gazebo 视觉实体 2s 内 delete+respawn 到新位置, 和 ROS 场保持同步
  - 顶部显示机器人当前采样值 /hazard/sample (实时)

线程模型 (很重要别改):
  - tkinter mainloop 必须跑在主线程
  - rclpy spin 跑在后台守护线程, GUI 和 spin 线程之间只通过 tk.StringVar
    做只读显示; Apply 按钮里调用的 call_async 通过 done_callback 返回结果,
    不会阻塞 GUI.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Float32
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


SOURCE_TYPES = ('fire', 'gas', 'pollution')
ALGORITHMS = ('geodesic', 'diffusion')


def _make_param(name: str, value) -> Parameter:
    """按 python 类型推断 ParameterValue, 只支持本面板用到的几种。"""
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
    """ROS 侧的轻量节点: 维护两个 set_parameters 客户端 + 订阅采样话题。"""

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
        """
        异步下发参数到两个目标节点. on_result(msg) 会在完成后被调用 (success / 错误信息).
        """
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


def _run_gui(node: ControlPanelNode) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title('Hazard Source Control Panel')
    root.geometry('360x340')

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill='both', expand=True)

    ttk.Label(frame, text='Source X (m)').grid(row=0, column=0, sticky='w')
    var_x = tk.StringVar(value='6.0')
    ttk.Entry(frame, textvariable=var_x, width=12).grid(row=0, column=1, sticky='we')

    ttk.Label(frame, text='Source Y (m)').grid(row=1, column=0, sticky='w')
    var_y = tk.StringVar(value='4.0')
    ttk.Entry(frame, textvariable=var_y, width=12).grid(row=1, column=1, sticky='we')

    ttk.Label(frame, text='Strength C0').grid(row=2, column=0, sticky='w')
    var_c0 = tk.StringVar(value='600.0')
    ttk.Entry(frame, textvariable=var_c0, width=12).grid(row=2, column=1, sticky='we')

    ttk.Label(frame, text='Source Type').grid(row=3, column=0, sticky='w')
    var_type = tk.StringVar(value='fire')
    ttk.Combobox(frame, textvariable=var_type, values=list(SOURCE_TYPES),
                 state='readonly', width=10).grid(row=3, column=1, sticky='we')

    ttk.Label(frame, text='Algorithm').grid(row=4, column=0, sticky='w')
    var_algo = tk.StringVar(value='geodesic')
    ttk.Combobox(frame, textvariable=var_algo, values=list(ALGORITHMS),
                 state='readonly', width=10).grid(row=4, column=1, sticky='we')

    ttk.Separator(frame).grid(row=5, column=0, columnspan=2, sticky='we', pady=6)

    ttk.Label(frame, text='Robot sample (/hazard/sample):').grid(
        row=6, column=0, columnspan=2, sticky='w')
    var_sample = tk.StringVar(value='(waiting...)')
    lbl_sample = ttk.Label(frame, textvariable=var_sample,
                           foreground='#007700', font=('TkDefaultFont', 12, 'bold'))
    lbl_sample.grid(row=7, column=0, columnspan=2, sticky='we', pady=2)

    var_status = tk.StringVar(value='')
    lbl_status = ttk.Label(frame, textvariable=var_status, foreground='#555555')
    lbl_status.grid(row=9, column=0, columnspan=2, sticky='we', pady=4)

    def on_apply():
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

    btn = ttk.Button(frame, text='Apply to ROS', command=on_apply)
    btn.grid(row=8, column=0, columnspan=2, sticky='we', pady=6)

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
