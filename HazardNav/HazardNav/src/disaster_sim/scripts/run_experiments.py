#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments.py
-----------------

批量自动化实验脚本：针对 hazard_source_node + gradient_explorer_node
构成的全系统管线，按场景矩阵批量运行仿真实验，自动收集 CSV 轨迹数据，
并生成统计报告（表格 + 路径长度 / 到达时间 / 成功率等指标）。

使用前提：
  - colcon build && source install/setup.bash 之后执行
  - 需要在有 Gazebo + ROS 2 的环境中运行（不能纯离线测试）
  - install 目录结构为 <ws>/install/disaster_sim/share/disaster_sim

用法：
    # 完整矩阵（所有场景各跑 3 次）
    python3 run_experiments.py \
        --scenarios source_positions.csv \
        --runs 3 \
        --output-dir ~/experiments/hazard_nav_$(date +%Y%m%d_%H%M%S) \
        --timeout 120

    # 快速冒烟测试（只跑默认源位置）
    python3 run_experiments.py --quick --runs 2

    # 查看帮助
    python3 run_experiments.py --help

场景矩阵格式（CSV）：
    source_type,source_x,source_y,algorithm,C0,alpha,arrival_thresh,label
    fire,6.0,4.0,geodesic,600.0,0.5,350.0,fire_center
    gas,2.0,2.0,diffusion,800.0,0.4,500.0,gas_corner
    (空行和 # 开头的行为注释)

生成的报告：
    output_dir/
    ├── run_001_fire_center/
    │   ├── run_001_fire_center_run1.csv
    │   ├── run_001_fire_center_run2.csv
    │   ├── run_001_fire_center_run3.csv
    │   └── summary.json
    ├── run_002_gas_corner/
    │   └── ...
    ├── all_scenarios_summary.csv     # 横向对比所有场景
    └── report.md                    # Markdown 格式统计报告
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """一次实验场景配置。"""
    source_type: str
    source_x: float
    source_y: float
    algorithm: str
    C0: float
    alpha: float
    arrival_thresh: float
    label: str

    @property
    def name(self) -> str:
        return self.label or f"{self.source_type}_{self.source_x:.1f}_{self.source_y:.1f}"


@dataclass
class RunResult:
    """单次实验运行的结果摘要。"""
    scenario_name: str
    run_id: int
    csv_path: str
    success: bool
    error: Optional[str]
    # 事后从 CSV 计算的指标
    travel_time_s: Optional[float]
    final_dist_to_source_m: Optional[float]
    min_dist_to_source_m: Optional[float]
    max_sample: Optional[float]
    arrived: bool
    path_length_m: Optional[float]
    avg_speed_mps: Optional[float]


@dataclass
class ScenarioStats:
    """一个场景多轮统计。"""
    scenario: Scenario
    runs: List[RunResult]
    success_count: int = 0
    success_rate: float = 0.0
    travel_time_mean_s: float = 0.0
    travel_time_std_s: float = 0.0
    final_dist_mean_m: float = 0.0
    final_dist_std_m: float = 0.0
    path_length_mean_m: float = 0.0
    path_length_std_m: float = 0.0


# ---------------------------------------------------------------------------
# CSV analysis
# ---------------------------------------------------------------------------

def analyse_csv(csv_path: str, source_x: float, source_y: float) -> Dict:
    """
    读取一条轨迹 CSV，计算关键指标。
    CSV 列: timestamp_sec, x, y, source_x, source_y,
             dist_to_source, sample, source_type, arrived
    """
    if not os.path.exists(csv_path):
        return dict(success=False, error="CSV not found")

    try:
        rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            return dict(success=False, error="CSV is empty")

        timestamps = [float(r['timestamp_sec']) for r in rows]
        xs = [float(r['x']) for r in rows]
        ys = [float(r['y']) for r in rows]
        dists = [float(r['dist_to_source']) for r in rows]
        samples = [float(r['sample']) for r in rows]
        arrived = int(rows[-1]['arrived']) if rows else 0

        travel_time = timestamps[-1] - timestamps[0]
        final_dist = dists[-1]
        min_dist = min(dists)
        max_sample = max(samples)

        # 路径总长度（累加相邻点欧氏距离）
        path_length = 0.0
        for i in range(1, len(xs)):
            path_length += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])

        avg_speed = path_length / travel_time if travel_time > 0 else 0.0

        return dict(
            success=True,
            travel_time_s=round(travel_time, 3),
            final_dist_m=round(final_dist, 4),
            min_dist_m=round(min_dist, 4),
            max_sample=round(max_sample, 4),
            arrived=bool(arrived),
            path_length_m=round(path_length, 4),
            avg_speed_mps=round(avg_speed, 4),
            num_points=len(rows),
        )
    except Exception as e:
        return dict(success=False, error=str(e))


# ---------------------------------------------------------------------------
# ROS 2 launch wrapper
# ---------------------------------------------------------------------------

def build_ros2_launch_cmd(
    scenario: Scenario,
    csv_path: str,
    output_dir: str,
    timeout: int,
) -> List[str]:
    """
    构造 ros2 launch 命令行。
    环境变量 COLCON_CURRENT_PREFIX 指向 install 目录时直接用 ros2 launch；
    否则从 install_dir 参数推断。
    """
    install_root = os.environ.get('COLCON_CURRENT_PREFIX', '')
    if not install_root:
        # 尝试从本脚本位置推断: <ws>/install/disaster_sim/share/disaster_sim -> <ws>
        candidate = Path(__file__).resolve().parents[4] / 'install'
        if candidate.exists():
            install_root = str(candidate)
        else:
            # 回退: 尝试标准 colcon build 位置
            ws = Path.cwd()
            cand = ws / 'install'
            if cand.exists():
                install_root = str(cand)

    pkg_share = os.path.join(install_root, 'disaster_sim', 'share', 'disaster_sim')
    params_file = os.path.join(pkg_share, 'config', 'hazard_params.yaml')

    cmd = [
        'ros2', 'launch', 'disaster_sim', 'disaster_nav.launch.py',
        f'source_type:={scenario.source_type}',
        f'source_x:={scenario.source_x}',
        f'source_y:={scenario.source_y}',
        f'algorithm:={scenario.algorithm}',
        f'C0:={scenario.C0}',
        f'arrival_thresh:={scenario.arrival_thresh}',
        f'trajectory_csv:={csv_path}',
        f'trajectory_dir:=',          # 空串使 trajectory_csv 优先
        f'log_trajectory:=true',
        f'use_rviz:=false',
        f'use_gazebo_viz:=true',
        f'use_control_panel:=false',
        f'auto_clean:=true',
    ]
    return cmd


def run_single_experiment(
    scenario: Scenario,
    run_id: int,
    output_dir: str,
    timeout: int,
    verbose: bool = False,
) -> RunResult:
    """
    执行一次实验：启动 launch，等待到达或超时，杀死进程，解析 CSV。
    """
    scenario_dir = os.path.join(output_dir, f"run_{scenario.name}")
    os.makedirs(scenario_dir, exist_ok=True)

    csv_path = os.path.join(
        scenario_dir,
        f"{scenario.name}_run{run_id}.csv",
    )

    launch_cmd = build_ros2_launch_cmd(scenario, csv_path, output_dir, timeout)

    if verbose:
        cmd_str = ' '.join(launch_cmd)
        print(f"\n  [CMD] {cmd_str}")

    t_start = time.time()
    proc = None
    error_msg: Optional[str] = None

    try:
        # 启动 launch 进程
        proc = subprocess.Popen(
            launch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 等待到达或超时
        elapsed = 0.0
        check_interval = 5.0  # 每 5s 检查一次 CSV 是否出现 arrived=1
        last_check = 0.0

        while elapsed < timeout:
            time.sleep(1.0)
            elapsed = time.time() - t_start

            # 每 check_interval 秒检查一次 CSV
            if elapsed - last_check >= check_interval and os.path.exists(csv_path):
                last_check = elapsed
                try:
                    with open(csv_path, newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows and int(rows[-1].get('arrived', 0)) == 1:
                            if verbose:
                                print(f"  [ARRIVED] t={elapsed:.1f}s, sample={rows[-1]['sample']}")
                            break
                except Exception:
                    pass

            # 检查进程是否异常退出
            if proc.poll() is not None:
                # 进程已退出但未到达，检查输出
                remaining = proc.stdout.read() if proc.stdout else ''
                if 'error' in remaining.lower():
                    error_msg = remaining[-500:]
                elif proc.returncode != 0:
                    error_msg = f"launch exited with code {proc.returncode}"
                break

        # 主动终止进程
        if proc.poll() is None:
            if verbose:
                print(f"  [KILL] timeout={timeout}s reached, terminating launch")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # 读取剩余输出（诊断用）
        stdout_tail = proc.stdout.read() if proc.stdout else '' if proc else ''
        if verbose and stdout_tail:
            print(f"  [LAUNCH OUTPUT TAIL]\n{stdout_tail[-1000:]}")

    except FileNotFoundError as e:
        error_msg = f"ros2 command not found: {e}"
    except Exception as e:
        error_msg = str(e)
        if proc and proc.poll() is None:
            proc.terminate()
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()

    wall_time = time.time() - t_start

    # 分析 CSV
    metrics = analyse_csv(csv_path, scenario.source_x, scenario.source_y)

    if not metrics.get('success', False):
        return RunResult(
            scenario_name=scenario.name,
            run_id=run_id,
            csv_path=csv_path,
            success=False,
            error=metrics.get('error', 'unknown'),
            travel_time_s=None,
            final_dist_to_source_m=None,
            min_dist_to_source_m=None,
            max_sample=None,
            arrived=False,
            path_length_m=None,
            avg_speed_mps=None,
        )

    return RunResult(
        scenario_name=scenario.name,
        run_id=run_id,
        csv_path=csv_path,
        success=True,
        error=None,
        travel_time_s=metrics.get('travel_time_s'),
        final_dist_to_source_m=metrics.get('final_dist_m'),
        min_dist_to_source_m=metrics.get('min_dist_m'),
        max_sample=metrics.get('max_sample'),
        arrived=metrics.get('arrived', False),
        path_length_m=metrics.get('path_length_m'),
        avg_speed_mps=metrics.get('avg_speed_mps'),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(scenario: Scenario, runs: List[RunResult]) -> ScenarioStats:
    """计算一个场景多轮实验的统计量。"""
    successful = [r for r in runs if r.success]
    n = len(successful)

    def _mean(key: str) -> float:
        return sum(getattr(r, key) for r in successful) / n if n > 0 else float('nan')

    def _std(key: str) -> float:
        if n < 2:
            return 0.0
        vals = [getattr(r, key) for r in successful]
        m = sum(vals) / n
        return math.sqrt(sum((v - m) ** 2 for v in vals) / n)

    s = ScenarioStats(scenario=scenario, runs=runs)
    if n > 0:
        s.success_count = n
        s.success_rate = n / len(runs)
        s.travel_time_mean_s = _mean('travel_time_s')
        s.travel_time_std_s = _std('travel_time_s')
        s.final_dist_mean_m = _mean('final_dist_to_source_m')
        s.final_dist_std_m = _std('final_dist_to_source_m')
        s.path_length_mean_m = _mean('path_length_m')
        s.path_length_std_m = _std('path_length_m')

    return s


def format_num(val: float, decimals: int = 3) -> str:
    if math.isnan(val) or math.isinf(val):
        return 'N/A'
    return f"{val:.{decimals}f}"


def write_markdown_report(
    all_stats: List[ScenarioStats],
    output_dir: str,
    args,
) -> str:
    """生成 Markdown 统计报告。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows_by_cat: Dict[str, List[ScenarioStats]] = {}
    for s in all_stats:
        cat = s.scenario.source_type
        rows_by_cat.setdefault(cat, []).append(s)

    lines = [
        "# HazardNav 实验报告",
        "",
        f"生成时间: {now}",
        f"实验配置: runs={args.runs}, timeout={args.timeout}s",
        f"场景数量: {len(all_stats)}",
        "",
        "---",
        "",
    ]

    # 汇总表
    lines += [
        "## 场景汇总",
        "",
        "| # | 场景名称 | 类型 | 源坐标 | 算法 | 成功率 | "
        "平均耗时(s) | 平均路径(m) | 最终距源(m) |",
        "|--:|---------|------|--------|------|--------|"
        "----------:|----------:|-----------:|",
    ]

    for i, s in enumerate(all_stats, 1):
        sc = s.scenario
        lines.append(
            f"| {i} | `{sc.label}` | {sc.source_type} | "
            f"({sc.source_x}, {sc.source_y}) | {sc.algorithm} | "
            f"{s.success_count}/{args.runs} ({s.success_rate:.0%}) | "
            f"{format_num(s.travel_time_mean_s)}±{format_num(s.travel_time_std_s)} | "
            f"{format_num(s.path_length_mean_m)}±{format_num(s.path_length_std_m)} | "
            f"{format_num(s.final_dist_mean_m)}±{format_num(s.final_dist_std_m)} |"
        )

    lines += ["", "---", ""]

    # 每类源的详细分析
    for cat, cat_stats in rows_by_cat.items():
        lines += [
            f"## {cat.upper()} 源详细分析",
            "",
            f"共 {len(cat_stats)} 个场景:",
            "",
        ]
        for s in cat_stats:
            sc = s.scenario
            lines += [
                f"### `{sc.label}` — ({sc.source_x}, {sc.source_y}), {sc.algorithm}",
                "",
                f"- C0={sc.C0}, alpha={sc.alpha}, arrival_thresh={sc.arrival_thresh}",
                f"- 成功率: {s.success_count}/{args.runs} = **{s.success_rate:.0%}**",
                f"- 到达耗时: **{format_num(s.travel_time_mean_s)} s** ± {format_num(s.travel_time_std_s)} s",
                f"- 路径长度: **{format_num(s.path_length_mean_m)} m** ± {format_num(s.path_length_std_m)} m",
                f"- 最终距源: **{format_num(s.final_dist_mean_m)} m** ± {format_num(s.final_dist_std_m)} m",
                "",
            ]
            # 各轮明细
            lines += [
                "| 轮次 | 到达 | 耗时(s) | 路径(m) | 最终距源(m) | 最短距源(m) | 最大读数 | 错误 |",
                "|----:|-----:|--------:|--------:|------------:|------------:|--------:|------|",
            ]
            for r in s.runs:
                err = r.error[:40] + '…' if r.error and len(r.error) > 40 else (r.error or '')
                lines.append(
                    f"| {r.run_id} | {'✓' if r.arrived else '✗'} | "
                    f"{format_num(r.travel_time_s)} | "
                    f"{format_num(r.path_length_m)} | "
                    f"{format_num(r.final_dist_to_source_m)} | "
                    f"{format_num(r.min_dist_to_source_m)} | "
                    f"{format_num(r.max_sample)} | "
                    f"{err} |"
                )
            lines.append("")

        lines += ["---", ""]

    # 参数表
    lines += [
        "## 实验参数",
        "",
        "| 参数 | 值 |",
        "|------|----|",
        f"| runs per scenario | {args.runs} |",
        f"| timeout per run (s) | {args.timeout} |",
        f"| source_type 列表 | {', '.join(sorted(set(s.scenario.source_type for s in all_stats)))} |",
        f"| algorithm 列表 | {', '.join(sorted(set(s.scenario.algorithm for s in all_stats)))} |",
        "",
    ]

    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return report_path


def write_csv_summary(
    all_stats: List[ScenarioStats],
    output_dir: str,
) -> str:
    """生成 all_scenarios_summary.csv。"""
    csv_path = os.path.join(output_dir, 'all_scenarios_summary.csv')
    header = [
        'label', 'source_type', 'source_x', 'source_y', 'algorithm',
        'C0', 'alpha', 'arrival_thresh',
        'runs', 'success_count', 'success_rate',
        'travel_time_mean_s', 'travel_time_std_s',
        'path_length_mean_m', 'path_length_std_m',
        'final_dist_mean_m', 'final_dist_std_m',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in all_stats:
            sc = s.scenario
            w.writerow([
                sc.label, sc.source_type, sc.source_x, sc.source_y, sc.algorithm,
                sc.C0, sc.alpha, sc.arrival_thresh,
                len(s.runs), s.success_count, f"{s.success_rate:.4f}",
                f"{s.travel_time_mean_s:.4f}", f"{s.travel_time_std_s:.4f}",
                f"{s.path_length_mean_m:.4f}", f"{s.path_length_std_m:.4f}",
                f"{s.final_dist_mean_m:.4f}", f"{s.final_dist_std_m:.4f}",
            ])
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_scenarios_from_csv(csv_path: str) -> List[Scenario]:
    """从 CSV 文件解析场景列表。"""
    scenarios = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get('label', '').strip() or None
            if label and label.startswith('#'):
                continue
            scenarios.append(Scenario(
                source_type=str(row['source_type']).strip(),
                source_x=float(row['source_x']),
                source_y=float(row['source_y']),
                algorithm=str(row.get('algorithm', 'geodesic')).strip(),
                C0=float(row.get('C0', 600.0)),
                alpha=float(row.get('alpha', 0.5)),
                arrival_thresh=float(row.get('arrival_thresh', 350.0)),
                label=label or '',
            ))
    return scenarios


def default_scenarios() -> List[Scenario]:
    """内置的默认场景矩阵，覆盖常见配置组合。"""
    return [
        # fire + geodesic
        Scenario('fire', 6.0, 4.0, 'geodesic', 600.0, 0.5, 350.0, 'fire_center_geo'),
        Scenario('fire', 2.0, 2.0, 'geodesic', 600.0, 0.5, 350.0, 'fire_corner_geo'),
        Scenario('fire', 10.0, 8.0, 'geodesic', 600.0, 0.5, 350.0, 'fire_far_geo'),
        # fire + diffusion
        Scenario('fire', 6.0, 4.0, 'diffusion', 600.0, 0.5, 350.0, 'fire_center_dif'),
        Scenario('fire', 2.0, 2.0, 'diffusion', 600.0, 0.5, 350.0, 'fire_corner_dif'),
        # gas + geodesic
        Scenario('gas', 6.0, 4.0, 'geodesic', 800.0, 0.4, 500.0, 'gas_center_geo'),
        Scenario('gas', 2.0, 2.0, 'geodesic', 800.0, 0.4, 500.0, 'gas_corner_geo'),
        # pollution + diffusion
        Scenario('pollution', 6.0, 4.0, 'diffusion', 700.0, 0.45, 400.0, 'poll_center_dif'),
        # 窄通道（测试绕障）
        Scenario('fire', 9.0, 3.0, 'geodesic', 600.0, 0.5, 350.0, 'fire_corridor_geo'),
    ]


def main():
    parser = argparse.ArgumentParser(
        description='HazardNav 批量自动化实验脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--scenarios', type=str, default='',
        help='CSV 文件路径，含场景矩阵；空则使用内置默认矩阵'
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='每个场景的重复次数（默认 3）'
    )
    parser.add_argument(
        '--output-dir', type=str, default='',
        help='实验输出根目录；空则自动生成带时间戳的目录'
    )
    parser.add_argument(
        '--timeout', type=int, default=120,
        help='单次实验超时时间，秒（默认 120s）'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='快速冒烟测试：只跑默认源位置，1 次'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='打印每轮实验的 launch 输出尾部'
    )
    args = parser.parse_args()

    # 解析场景
    if args.scenarios:
        scenarios = parse_scenarios_from_csv(args.scenarios)
        print(f"[SCENARIOS] 从 CSV 加载了 {len(scenarios)} 个场景")
    elif args.quick:
        scenarios = [Scenario('fire', 6.0, 4.0, 'geodesic', 600.0, 0.5, 350.0, 'quick_test')]
        args.runs = 1
        print("[SCENARIOS] 快速冒烟测试模式")
    else:
        scenarios = default_scenarios()
        print(f"[SCENARIOS] 使用内置默认矩阵，共 {len(scenarios)} 个场景")

    # 输出目录
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.expanduser('~'), f'hazard_nav_exp_{ts}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[OUTPUT] {output_dir}")

    # 保存配置快照
    config_snapshot = {
        'generated_at': datetime.now().isoformat(),
        'args': vars(args),
        'scenarios': [asdict(s) for s in scenarios],
    }
    with open(os.path.join(output_dir, 'experiment_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    all_stats: List[ScenarioStats] = []
    total_runs = len(scenarios) * args.runs
    run_counter = 0

    t_total_start = time.time()

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"[SCENARIO] {scenario.name} | {scenario.source_type} | "
              f"({scenario.source_x}, {scenario.source_y}) | {scenario.algorithm}")
        print(f"{'='*60}")

        scenario_dir = os.path.join(output_dir, f"run_{scenario.name}")
        os.makedirs(scenario_dir, exist_ok=True)

        runs: List[RunResult] = []
        for run_id in range(1, args.runs + 1):
            run_counter += 1
            t_run_start = time.time()
            print(f"\n  [RUN {run_id}/{args.runs}] ({run_counter}/{total_runs}) ...", end='', flush=True)

            result = run_single_experiment(
                scenario, run_id, output_dir, args.timeout, args.verbose,
            )
            runs.append(result)

            elapsed = time.time() - t_run_start
            if result.success:
                arrived_str = 'ARRIVED' if result.arrived else 'TIMEOUT'
                print(f" {elapsed:.0f}s [{arrived_str}] "
                      f"travel={format_num(result.travel_time_s)}s "
                      f"dist={format_num(result.final_dist_to_source_m)}m")
            else:
                print(f" {elapsed:.0f}s [FAIL] {result.error}")

        # 写入单场景 summary.json
        scenario_stats = compute_stats(scenario, runs)
        all_stats.append(scenario_stats)
        with open(os.path.join(scenario_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'scenario': asdict(scenario),
                'stats': asdict(scenario_stats),
                'runs': [asdict(r) for r in runs],
            }, f, indent=2, ensure_ascii=False, default=lambda x: None)

        # 打印该场景摘要
        s = scenario_stats
        print(f"\n  [SUMMARY] {scenario.name}: "
              f"success={s.success_count}/{args.runs} "
              f"time={format_num(s.travel_time_mean_s)}±{format_num(s.travel_time_std_s)}s "
              f"dist={format_num(s.final_dist_mean_m)}±{format_num(s.final_dist_std_m)}m")

    # 生成报告
    total_time = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"[ALL DONE] {total_runs} runs in {total_time:.0f}s "
          f"({total_time / total_runs:.1f}s/run avg)")

    csv_path = write_csv_summary(all_stats, output_dir)
    print(f"[CSV] {csv_path}")

    report_path = write_markdown_report(all_stats, output_dir, args)
    print(f"[REPORT] {report_path}")
    print(f"[OUTPUT DIR] {output_dir}")


if __name__ == '__main__':
    main()
