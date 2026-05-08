#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disaster_nav.launch.py
----------------------

一键启动: Gazebo 差速仿真 + 原 SLAM/A*/纯追踪 + 灾害扩散仿真
       + Gazebo 内灾害源可视化 + 梯度引导探索 + RViz.

等价于以前需要手动依次执行的三条命令:
  1) ros2 launch gazebo_modele gazebo.launch.py
  2) ros2 launch nav_slam 2dpoints.launch.py
  3) (手动在 RViz 里点 2D Goal Pose)

整合成单条:
  ros2 launch disaster_sim disaster_nav.launch.py \
       source_type:=fire source_x:=6.0 source_y:=4.0 algorithm:=geodesic

参数 (全部带默认值, 可命令行覆盖):
  source_type       fire | gas | pollution              默认 fire
  source_x          源的 x 坐标 (m)                     默认 6.0
  source_y          源的 y 坐标 (m)                     默认 4.0
  algorithm         geodesic | diffusion                默认 geodesic
  C0                源强度 (℃ / ppm / μg·m⁻³)           默认 600.0
  arrival_thresh    采样触发"到达"的读数阈值            默认 350.0
                    * 该值必须和 (C0, alpha) 匹配,
                      否则会"一启动就判到达"卡原地, 详见 hazard_params.yaml
  use_rviz          是否启动 RViz                       默认 true
  use_gazebo_viz    是否在 Gazebo 里 spawn 灾害源实体   默认 true
  use_control_panel 是否打开 tkinter 控制面板           默认 true (无头环境改 false)
  log_trajectory    是否把轨迹写 CSV                    默认 true
  trajectory_csv    CSV 路径; 空串走 trajectory_dir + 时间戳自动命名
  trajectory_dir    空串/默认自动指向 <workspace>/src/csv/ (若存在), 否则 ~/.
                    这是 "CSV 集中归档" 目录, 每次 launch 都会在这里新生成
                    一份 disaster_trajectory_<YYYYmmdd_HHMMSS>.csv.

  auto_clean        启动前自动 pkill 残留 gzserver/gzclient/nav_slam/disaster_sim
                    节点, 再延迟 ~3s 启动 Gazebo. 默认 true (强烈建议).
                    关掉用 auto_clean:=false.

--- 关于残留进程 ---
gazebo 偶发不能被 SIGINT 优雅关闭 (X/OpenGL 崩溃时尤甚). 如果你 Ctrl+C
之后再次启动看到 "Address already in use" / "entity already exists" /
topic 上多个 publisher, 大概率是残留. 默认 auto_clean=true 已经会在启动前
执行以下清理, 通常无需手动干预:
    pkill -KILL -u "$USER" -f gzserver
    pkill -KILL -u "$USER" -f gzclient
    pkill -KILL -u "$USER" -f "disaster_sim/lib"
    pkill -KILL -u "$USER" -f "nav_slam/lib"
    pkill -KILL -u "$USER" -f "gazebo --verbose"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    GroupAction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


PRE_CLEAN_CMD = (
    'echo "[disaster_nav] auto_clean=true, killing stale gazebo/ros nodes..."; '
    'pkill -KILL -u "$USER" -f gzserver 2>/dev/null; '
    'pkill -KILL -u "$USER" -f gzclient 2>/dev/null; '
    'pkill -KILL -u "$USER" -f "disaster_sim/lib" 2>/dev/null; '
    'pkill -KILL -u "$USER" -f "nav_slam/lib" 2>/dev/null; '
    'pkill -KILL -u "$USER" -f "gazebo --verbose" 2>/dev/null; '
    'sleep 2; '
    'echo "[disaster_nav] pre-clean done."; '
    'true'
)


def _guess_workspace_csv_dir(share_dir: str) -> str:
    """
    根据 install/disaster_sim/share/disaster_sim 目录反推工作区根 + src/csv/.
    典型路径:
        <ws>/install/disaster_sim/share/disaster_sim
    往上 3 级 -> <ws>/install -> 再往上 1 级 -> <ws>, 拼 src/csv.
    若该目录存在就用它, 否则返回空串 (logger 端会回退到 ~/).
    """
    try:
        ws_root = os.path.abspath(
            os.path.join(share_dir, '..', '..', '..', '..'))
        cand = os.path.join(ws_root, 'src', 'csv')
        if os.path.isdir(cand):
            return cand
    except Exception:
        pass
    return ''


def generate_launch_description():
    pkg_disaster = get_package_share_directory('disaster_sim')
    pkg_gazebo = get_package_share_directory('gazebo_modele')
    get_package_share_directory('nav_slam')  # 只校验存在

    default_params = os.path.join(pkg_disaster, 'config', 'hazard_params.yaml')
    default_rviz = os.path.join(pkg_disaster, 'config', 'rviz_hazard.rviz')
    default_csv_dir = _guess_workspace_csv_dir(pkg_disaster)

    source_type = LaunchConfiguration('source_type')
    source_x = LaunchConfiguration('source_x')
    source_y = LaunchConfiguration('source_y')
    algorithm = LaunchConfiguration('algorithm')
    C0 = LaunchConfiguration('C0')
    arrival_thresh = LaunchConfiguration('arrival_thresh')
    use_rviz = LaunchConfiguration('use_rviz')
    use_gazebo_viz = LaunchConfiguration('use_gazebo_viz')
    use_control_panel = LaunchConfiguration('use_control_panel')
    log_trajectory = LaunchConfiguration('log_trajectory')
    trajectory_csv = LaunchConfiguration('trajectory_csv')
    trajectory_dir = LaunchConfiguration('trajectory_dir')
    params_file = LaunchConfiguration('params_file')
    rviz_config = LaunchConfiguration('rviz_config')
    auto_clean = LaunchConfiguration('auto_clean')

    args = [
        DeclareLaunchArgument('source_type', default_value='fire',
                              description='fire | gas | pollution'),
        DeclareLaunchArgument('source_x', default_value='6.0',
                              description='hazard source x in map frame (m)'),
        DeclareLaunchArgument('source_y', default_value='4.0',
                              description='hazard source y in map frame (m)'),
        DeclareLaunchArgument('algorithm', default_value='geodesic',
                              description='geodesic | diffusion'),
        DeclareLaunchArgument('C0', default_value='600.0',
                              description='source strength; unit depends on source_type'),
        DeclareLaunchArgument('arrival_thresh', default_value='350.0',
                              description='sample threshold that triggers STOP. '
                                          'must be chosen together with C0/alpha, '
                                          'see hazard_params.yaml for the arrival geometry.'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_gazebo_viz', default_value='true',
                              description='spawn a visual entity for the hazard source in Gazebo'),
        DeclareLaunchArgument('use_control_panel', default_value='true',
                              description='open the tkinter GUI to tweak source x/y/C0/type/algorithm live'),
        DeclareLaunchArgument('log_trajectory', default_value='true',
                              description='log (x, y, dist_to_source, sample) to a CSV file'),
        DeclareLaunchArgument('trajectory_csv', default_value='',
                              description='CSV path; empty -> <trajectory_dir>/disaster_trajectory_<timestamp>.csv'),
        DeclareLaunchArgument('trajectory_dir', default_value=default_csv_dir,
                              description='Directory to place auto-named CSVs when trajectory_csv is empty. '
                                          'Defaults to <workspace>/src/csv/ if that folder exists, '
                                          'otherwise ~/ (home dir).'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument(
            'auto_clean', default_value='true',
            description='pkill stale gazebo/ros nodes before launch, then delay '
                        '~3s so port 11345 is freed. Set false to skip.'),
    ]

    pre_clean = ExecuteProcess(
        cmd=['bash', '-c', PRE_CLEAN_CMD],
        output='screen',
        condition=IfCondition(auto_clean),
    )

    # 1) 原差速仿真: Gazebo + URDF + 基础 TF (复用原项目 launch, 不改动其内部)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo, 'launch', 'gazebo.launch.py')),
    )

    # 2) 原导航管线: astar / map_pub / odom_map_tf / points_pub_map / start_nav
    nav_nodes = GroupAction(actions=[
        Node(package='nav_slam', executable='astar',
             name='astar', output='screen', respawn=False),
        Node(package='nav_slam', executable='map_pub',
             name='map_pub', output='screen', respawn=False),
        Node(package='nav_slam', executable='odom_map_tf',
             name='odom_map_tf', output='screen', respawn=False),
        Node(package='nav_slam', executable='points_pub_map',
             name='points_pub_map', output='screen', respawn=False),
        Node(package='nav_slam', executable='start_nav',
             name='start_nav', output='screen', respawn=False),
    ])

    # 3) 新增: 灾害源扩散仿真 (发布 /hazard/field, /hazard/sample, Marker, 源真值)
    hazard_node = Node(
        package='disaster_sim',
        executable='hazard_source_node',
        name='hazard_source_node',
        output='screen',
        respawn=False,
        parameters=[
            params_file,
            {
                'source_type': source_type,
                'source_x': source_x,
                'source_y': source_y,
                'algorithm': algorithm,
                'C0': C0,
            },
        ],
    )

    # 4) 新增: 梯度引导探索 (替代 RViz 手动点 /goal_pose)
    #    纯场驱动: 不订阅 /hazard/source_gt 进行导航，仅依靠 /hazard/field
    #    的场值在膨胀后的 /combined_grid 可达域上 BFS 最大化场值，自动趋近源头。
    explorer_node = Node(
        package='disaster_sim',
        executable='gradient_explorer_node',
        name='gradient_explorer_node',
        output='screen',
        respawn=False,
        parameters=[
            params_file,
            {
                'arrival_sample_threshold': arrival_thresh,
                'use_source_goal': False,
            },
        ],
    )

    # 5) 新增: Gazebo 内把灾害源 spawn 为可见实体 (与 /hazard/* 同位置)
    gazebo_viz_node = Node(
        package='disaster_sim',
        executable='hazard_gazebo_visual_node',
        name='hazard_gazebo_visual_node',
        output='screen',
        respawn=False,
        condition=IfCondition(use_gazebo_viz),
        parameters=[
            {
                'source_type': source_type,
                'source_x': source_x,
                'source_y': source_y,
            },
        ],
    )

    # 6) RViz (带灾害场 + 障碍 + 路径 图层)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
    )

    # 7) 新增: tkinter 控制面板 (实时改源参数; 可关)
    control_panel_node = Node(
        package='disaster_sim',
        executable='hazard_control_panel_node',
        name='hazard_control_panel_node',
        output='screen',
        respawn=False,
        condition=IfCondition(use_control_panel),
    )

    # 8) 新增: 轨迹 CSV 记录器
    trajectory_logger_node = Node(
        package='disaster_sim',
        executable='trajectory_logger_node',
        name='trajectory_logger_node',
        output='screen',
        respawn=False,
        condition=IfCondition(log_trajectory),
        parameters=[
            {
                'output_csv': trajectory_csv,
                'output_dir': trajectory_dir,
                'source_type': source_type,
                'fallback_source_x': source_x,
                'fallback_source_y': source_y,
                'arrival_sample_threshold': arrival_thresh,
            },
        ],
    )

    # 延迟启动: 给 pre_clean 的 pkill + sleep 留出时间释放 11345 端口.
    # 即使 auto_clean=false, 延迟几秒也更稳 (旧进程自然退出).
    delayed_gazebo = TimerAction(period=3.0, actions=[gazebo_launch])
    delayed_nav = TimerAction(period=3.5, actions=[nav_nodes])
    delayed_hazard = TimerAction(
        period=5.0,
        actions=[hazard_node, explorer_node, gazebo_viz_node],
    )
    delayed_rviz = TimerAction(period=5.5, actions=[rviz_node])
    delayed_panel = TimerAction(
        period=6.0,
        actions=[control_panel_node, trajectory_logger_node],
    )

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)
    ld.add_action(pre_clean)
    ld.add_action(delayed_gazebo)
    ld.add_action(delayed_nav)
    ld.add_action(delayed_hazard)
    ld.add_action(delayed_rviz)
    ld.add_action(delayed_panel)
    return ld
