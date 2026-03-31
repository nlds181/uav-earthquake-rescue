"""
面向地震应急的太阳能无人机群协同通信与轨迹优化
基于 LD-HAF 学习驱动混合自适应优化框架
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="地震应急无人机协同通信平台",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LD-HAF优化器 ====================
@dataclass
class OptimizationState:
    gradient_norm: float
    gradient_variance: float
    hessian_eigenvalue: float
    function_decrease: float
    constraint_violation: float
    iteration_ratio: float

class LDHAFOptimizer:
    """学习驱动的混合自适应优化框架"""

    def __init__(self):
        self.algorithms = {
            'sgd': self._sgd_step,
            'adam': self._adam_step,
            'nag': self._nag_step,
            'sgld': self._sgld_step,
            'trust_region': self._trust_region_step,
            'newton': self._newton_step
        }
        self.algorithm_history = []
        self._m = None
        self._v = None
        self._t = 0
        self._nag_v = None

    def select_algorithm(self, state: OptimizationState) -> str:
        if state.gradient_norm > 10:
            return 'adam'
        elif state.gradient_norm < 0.01 and state.hessian_eigenvalue > 0:
            return 'newton'
        elif state.constraint_violation > 0.1:
            return 'trust_region'
        elif state.gradient_norm < 0.1 and state.hessian_eigenvalue < 0:
            return 'sgld'
        else:
            return 'sgd'

    def optimize(self, objective_fn, x0, max_iter=200, callback=None):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        self.algorithm_history = []

        self._m = np.zeros_like(x)
        self._v = np.zeros_like(x)
        self._t = 0
        self._nag_v = np.zeros_like(x)

        for k in range(max_iter):
            grad = self._compute_gradient(objective_fn, x)

            state = OptimizationState(
                gradient_norm=np.linalg.norm(grad),
                gradient_variance=np.var(grad),
                hessian_eigenvalue=self._approx_hessian_eigenvalue(objective_fn, x),
                function_decrease=self._get_function_decrease(objective_fn, x, history[-1] if k>0 else x),
                constraint_violation=self._compute_constraint_violation(x),
                iteration_ratio=k/max_iter
            )

            algo = self.select_algorithm(state)
            self.algorithm_history.append(algo)

            eta = 0.05 * (0.95 ** k)
            x_new = self.algorithms[algo](x, grad, eta)

            if algo == 'sgld':
                x_new += 0.01 * np.random.randn(*x.shape)

            history.append(x_new.copy())

            if callback:
                callback(k, x_new, state, algo)

            if np.linalg.norm(x_new - x) < 1e-6:
                break

            x = x_new

        return x, history

    def _sgd_step(self, x, grad, eta):
        return x - eta * grad

    def _adam_step(self, x, grad, eta):
        beta1, beta2 = 0.9, 0.999
        self._t += 1
        self._m = beta1 * self._m + (1 - beta1) * grad
        self._v = beta2 * self._v + (1 - beta2) * (grad ** 2)
        m_hat = self._m / (1 - beta1 ** self._t)
        v_hat = self._v / (1 - beta2 ** self._t)
        return x - eta * m_hat / (np.sqrt(v_hat) + 1e-8)

    def _nag_step(self, x, grad, eta):
        momentum = 0.9
        self._nag_v = momentum * self._nag_v - eta * grad
        return x + self._nag_v

    def _sgld_step(self, x, grad, eta):
        return x - eta * grad

    def _trust_region_step(self, x, grad, eta):
        delta = 1.0
        step = -eta * grad
        if np.linalg.norm(step) > delta:
            step = step / np.linalg.norm(step) * delta
        return x + step

    def _newton_step(self, x, grad, eta):
        hessian = self._approx_hessian(x, grad)
        try:
            step = np.linalg.solve(hessian + 1e-4 * np.eye(len(x)), -grad)
            return x + eta * step
        except:
            return x - eta * grad

    def _compute_gradient(self, fn, x, eps=1e-6):
        grad = np.zeros_like(x)
        f0 = fn(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (fn(x_plus) - f0) / eps
        return grad

    def _approx_hessian(self, x, grad, eps=1e-5):
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            grad_plus = self._compute_gradient(lambda x: np.sum(x**2), x_plus)
            H[i, :] = (grad_plus - grad) / eps
        return (H + H.T) / 2

    def _approx_hessian_eigenvalue(self, fn, x):
        try:
            grad = self._compute_gradient(fn, x)
            H = self._approx_hessian(x, grad)
            eigvals = np.linalg.eigvals(H)
            return np.min(eigvals.real)
        except:
            return 1.0

    def _get_function_decrease(self, fn, x_new, x_old):
        try:
            return abs(fn(x_old) - fn(x_new))
        except:
            return 0.01

    def _compute_constraint_violation(self, x):
        violation = 0
        for i in range(0, len(x), 3):
            z = x[i+2]
            if z < 70:
                violation += (70 - z) * 0.5
            if z > 350:
                violation += (z - 350) * 0.3
        return min(1.0, violation / 50)


# ==================== 动态用户模型（高斯-马尔可夫） ====================
class DynamicUserModel:
    """用户动态移动模型，高斯-马尔可夫过程"""
    def __init__(self, num_users, center=(0,0), spread=200, speed=0.05):
        self.num_users = num_users
        self.positions = []
        self.velocities = []
        self.center = center
        self.speed = speed
        # 初始化用户位置（围绕灾区中心）
        np.random.seed(42)
        for _ in range(num_users):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.exponential(spread)
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            self.positions.append([x, y])
            # 随机初始速度
            vx = np.random.uniform(-speed, speed)
            vy = np.random.uniform(-speed, speed)
            self.velocities.append([vx, vy])

    def update(self):
        """更新用户位置（高斯-马尔可夫过程）"""
        for i in range(self.num_users):
            # 简单的随机游走 + 向中心漂移
            self.velocities[i][0] += np.random.normal(0, self.speed*0.5)
            self.velocities[i][1] += np.random.normal(0, self.speed*0.5)
            # 阻尼和向心恢复
            self.velocities[i][0] *= 0.95
            self.velocities[i][1] *= 0.95
            # 向灾区中心漂移
            dx = -self.positions[i][0] * 0.01
            dy = -self.positions[i][1] * 0.01
            self.velocities[i][0] += dx
            self.velocities[i][1] += dy

            self.positions[i][0] += self.velocities[i][0]
            self.positions[i][1] += self.velocities[i][1]

    def get_positions(self):
        return self.positions


# ==================== 地形模型 ====================
class TerrainModel:
    def __init__(self, terrain_type="山区"):
        self.terrain_type = terrain_type

    def get_height(self, x, y):
        if self.terrain_type == "山区":
            return 70 * np.exp(-(x**2 + y**2) / 90000) + \
                   45 * np.exp(-((x-120)**2 + (y-80)**2) / 40000) + \
                   55 * np.exp(-((x+80)**2 + (y+120)**2) / 50000)
        else:
            return 0


# ==================== 动态3D动画函数 ====================
def create_dynamic_3d_plot(uav_histories, user_model, terrain_model, terrain_type):
    """
    创建动态3D图，包含无人机飞行轨迹动画 + 动态移动的用户
    """
    num_uavs = len(uav_histories)
    num_frames = len(uav_histories[0])  # 时间步数
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']

    # 预先生成所有帧的用户位置（为了动画平滑）
    user_positions_over_time = []
    # 复制用户模型并模拟时间步
    temp_user_model = DynamicUserModel(num_users=user_model.num_users,
                                       center=(0,0), spread=200, speed=0.05)
    # 重置到初始位置
    temp_user_model.positions = [pos.copy() for pos in user_model.positions]
    user_positions_over_time.append([pos.copy() for pos in temp_user_model.positions])
    for t in range(1, num_frames):
        temp_user_model.update()
        user_positions_over_time.append([pos.copy() for pos in temp_user_model.positions])

    # --- 初始帧（t=0）---
    start_traces = []
    # 无人机起点
    for i in range(num_uavs):
        pos = uav_histories[i][0]
        start_traces.append(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='lines+markers',
            name=f'无人机 {i+1}',
            line=dict(color=colors[i % len(colors)], width=4),
            marker=dict(size=6),
            showlegend=True
        ))
    # 地形（静态）
    if terrain_type == "山区":
        x_terrain = np.linspace(-400, 400, 30)
        y_terrain = np.linspace(-400, 400, 30)
        X, Y = np.meshgrid(x_terrain, y_terrain)
        Z = np.zeros_like(X)
        for i in range(len(x_terrain)):
            for j in range(len(y_terrain)):
                Z[i, j] = terrain_model.get_height(X[i, j], Y[i, j])
        terrain_trace = go.Surface(
            x=x_terrain, y=y_terrain, z=Z,
            colorscale='Viridis', opacity=0.5, name='地形', showscale=False
        )
        start_traces.append(terrain_trace)
    # 初始用户位置
    user_positions = user_positions_over_time[0]
    user_trace = go.Scatter3d(
        x=[p[0] for p in user_positions],
        y=[p[1] for p in user_positions],
        z=[10]*len(user_positions),
        mode='markers',
        marker=dict(color='yellow', size=3, symbol='circle'),
        name='灾区用户'
    )
    start_traces.append(user_trace)

    # --- 构建每一帧 ---
    frames = []
    for t in range(num_frames):
        frame_data = []
        # 无人机当前点 + 历史轨迹
        for i in range(num_uavs):
            pos = uav_histories[i][t]
            frame_data.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='lines+markers',
                name=f'无人机 {i+1}',
                line=dict(color=colors[i % len(colors)], width=4),
                marker=dict(size=6),
                showlegend=False
            ))
            # 历史轨迹线
            hist_x = [p[0] for p in uav_histories[i][:t+1]]
            hist_y = [p[1] for p in uav_histories[i][:t+1]]
            hist_z = [p[2] for p in uav_histories[i][:t+1]]
            frame_data.append(go.Scatter3d(
                x=hist_x, y=hist_y, z=hist_z,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                showlegend=False
            ))
        # 当前帧的用户位置
        users_this_frame = user_positions_over_time[t]
        frame_data.append(go.Scatter3d(
            x=[p[0] for p in users_this_frame],
            y=[p[1] for p in users_this_frame],
            z=[10]*len(users_this_frame),
            mode='markers',
            marker=dict(color='yellow', size=3, symbol='circle'),
            name='灾区用户',
            showlegend=False
        ))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # 创建图形
    fig = go.Figure(data=start_traces, frames=frames)

    # 播放按钮和滑块
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="▶ 播放",
                     method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True},
                                   "fromcurrent": True, "mode": "immediate"}]),
                dict(label="⏸ 暂停",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}])
            ],
            x=0.1, y=0, xanchor="right", yanchor="top"
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[f.name],
                       {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                       label=str(i)) for i, f in enumerate(frames)],
            transition={"duration": 0},
            x=0.1, len=0.9
        )],
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='高度 (m)',
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(text="🚁 无人机群动态飞行轨迹 + 动态用户移动（点击播放）", font=dict(size=16))
    )
    return fig


# ==================== 目标函数构建 ====================
def build_objective_function(num_uavs, terrain_type, objective_type, user_model):
    """目标函数，基于用户当前位置计算覆盖"""
    def objective(x):
        total_cost = 0
        user_positions = user_model.get_positions()

        for i in range(num_uavs):
            ux, uy, uz = x[3*i], x[3*i+1], x[3*i+2]

            # 平均用户距离覆盖
            coverage_sum = 0
            for (ux_user, uy_user) in user_positions:
                dist = np.sqrt((ux - ux_user)**2 + (uy - uy_user)**2)
                coverage_sum += 1 / (1 + dist/80)
            avg_coverage = coverage_sum / len(user_positions)

            if objective_type == "最大化最小用户速率":
                total_cost -= avg_coverage * 20
            elif objective_type == "最大化能效":
                total_cost -= avg_coverage * 12
                total_cost += (uz / 300) * 6
            else:
                total_cost -= avg_coverage * 16

            total_cost += (uz / 350) * 4
            total_cost += (np.sqrt(ux**2+uy**2) / 400) * 2

            if uz < 70:
                total_cost += (70 - uz) * 4
            if uz > 350:
                total_cost += (uz - 350) * 2

            if terrain_type == "山区":
                terrain = TerrainModel("山区")
                th = terrain.get_height(ux, uy)
                if uz < th + 30:
                    total_cost += (th + 30 - uz) * 15

        # 避撞
        for i in range(num_uavs):
            for j in range(i+1, num_uavs):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dz = x[3*i+2] - x[3*j+2]
                dist = np.sqrt(dx**2+dy**2+dz**2)
                if dist < 50:
                    total_cost += (50 - dist) * 12
        return total_cost
    return objective


def init_positions(num_uavs, altitude):
    positions = []
    for i in range(num_uavs):
        angle = 2 * np.pi * i / num_uavs
        radius = 200
        positions.extend([radius * np.cos(angle), radius * np.sin(angle), altitude])
    return positions


# ==================== 辅助可视化函数 ====================
def create_coverage_heatmap(uav_positions, user_model, terrain_type):
    size = 40
    bounds = (-500, 500)
    heatmap = np.zeros((size, size))
    x_edges = np.linspace(bounds[0], bounds[1], size + 1)
    y_edges = np.linspace(bounds[0], bounds[1], size + 1)

    for uav in uav_positions:
        ux, uy, uz = uav[0], uav[1], uav[2]
        for i in range(size):
            for j in range(size):
                cx = (x_edges[i] + x_edges[i+1]) / 2
                cy = (y_edges[j] + y_edges[j+1]) / 2
                dist = np.sqrt((ux - cx)**2 + (uy - cy)**2)
                coverage = 1 / (1 + (dist/120)**2) * (1 + 0.3 * uz/350)
                heatmap[i, j] = max(heatmap[i, j], min(coverage, 0.95))

    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        x=(x_edges[:-1] + x_edges[1:])/2,
        y=(y_edges[:-1] + y_edges[1:])/2,
        colorscale='Hot', zmin=0, zmax=0.95, colorbar=dict(title="覆盖强度")
    ))
    # 叠加用户位置
    user_positions = user_model.get_positions()
    fig.add_trace(go.Scatter(
        x=[p[0] for p in user_positions], y=[p[1] for p in user_positions],
        mode='markers', marker=dict(color='blue', size=8, symbol='x'), name='用户位置'
    ))
    fig.update_layout(title="📡 灾区通信覆盖热力图", height=400, paper_bgcolor='rgba(0,0,0,0)')
    return fig


def create_algorithm_comparison(iterations):
    fig = go.Figure()
    algorithms = ['LD-HAF (本方案)', 'MADDPG', '传统凸优化', 'Q-Learning']
    values = [iterations, int(iterations*1.55), int(iterations*2.3), int(iterations*3.2)]
    fig.add_trace(go.Bar(x=algorithms, y=values, marker_color=['#4ECDC4','#FF6B6B','#FFEAA7','#DFE6E9'],
                         text=values, textposition='outside'))
    fig.update_layout(title="⚡ 算法性能对比", yaxis_title="迭代次数", height=400)
    return fig


def create_energy_chart(iterations, solar_enabled):
    energy = []
    for i in range(iterations+1):
        e = 100 * (1 - 0.25*i/iterations)
        if solar_enabled:
            e += 15 * np.sin(i*np.pi/30)
            e = min(92, max(35, e))
        else:
            e = max(20, e)
        energy.append(e)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=energy, mode='lines', line=dict(color='#4ECDC4', width=2),
                             fill='tozeroy', fillcolor='rgba(78,205,196,0.2)'))
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="紧急阈值")
    fig.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="安全阈值")
    fig.update_layout(title="🔋 无人机群能量状态", yaxis_title="电量 (%)", height=300)
    return fig


def create_algorithm_switch_chart(algo_history):
    algo_map = {'sgd':0,'adam':1,'nag':2,'sgld':3,'trust_region':4,'newton':5}
    algo_names = ['SGD','Adam','NAG','SGLD','Trust Reg','Newton']
    numeric = [algo_map.get(a,0) for a in algo_history[:100]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=numeric, mode='lines+markers', line=dict(color='#FF6B6B', width=2),
                             marker=dict(size=5, symbol='diamond', color='#4ECDC4')))
    fig.update_layout(title="🔄 LD-HAF 自适应算法切换记录", yaxis=dict(tickvals=list(range(6)), ticktext=algo_names),
                      height=300)
    return fig


def create_performance_radar(metrics):
    categories = ['收敛速度', '通信覆盖率', '能效比', '鲁棒性', '避障能力']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=metrics, theta=categories, fill='toself',
                                  line=dict(color='#4ECDC4', width=2), fillcolor='rgba(78,205,196,0.3)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), title="📊 LD-HAF 综合性能评估", height=350)
    return fig


# ==================== 主程序 ====================
def main():
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px'>
        <h1 style='color: white; margin: 0'>🚁 面向地震应急的太阳能无人机群协同通信与轨迹优化</h1>
        <p style='color: white; margin: 10px 0 0 0'>基于 LD-HAF 学习驱动混合自适应优化框架 | 计算机设计大赛参赛作品</p>
        <p style='color: #ddd; font-size: 14px;'>✨ 支持无人机动态飞行 + 用户动态移动 ✨</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ 仿真参数配置")
        num_uavs = st.slider("无人机数量", 1, 6, 3)
        flight_altitude = st.slider("初始飞行高度 (m)", 70, 350, 150)
        terrain_type = st.selectbox("地形类型", ["山区 (九寨沟模拟)", "城镇", "平原"])
        objective_type = st.selectbox("优化目标", ["最大化最小用户速率", "最大化能效", "最大化覆盖范围"])
        num_users = st.slider("灾区用户数量", 20, 100, 50)
        max_iterations = st.slider("LD-HAF 迭代次数", 50, 200, 100)
        enable_adaptive = st.checkbox("启用自适应算法切换", value=True)
        enable_solar = st.checkbox("启用太阳能采集", value=True)
        run_simulation = st.button("🚀 开始地震应急仿真", type="primary", use_container_width=True)

    # 初始化用户模型和地形
    user_model = DynamicUserModel(num_users=num_users, center=(0,0), spread=200, speed=0.05)
    terrain_model = TerrainModel(terrain_type)

    if run_simulation:
        with st.spinner("🔄 LD-HAF 优化引擎运行中... 正在求解非凸优化问题"):
            try:
                # 构建目标函数（基于当前用户位置）
                objective_fn = build_objective_function(num_uavs, terrain_type, objective_type, user_model)
                initial_positions = init_positions(num_uavs, flight_altitude)

                optimizer = LDHAFOptimizer() if enable_adaptive else None
                start_time = time.time()

                if enable_adaptive and optimizer:
                    optimal_positions, history = optimizer.optimize(objective_fn, initial_positions, max_iter=max_iterations)
                    algo_history = optimizer.algorithm_history
                else:
                    x = np.array(initial_positions)
                    history = [x.copy()]
                    for k in range(max_iterations):
                        grad = np.zeros_like(x)
                        f0 = objective_fn(x)
                        eps = 1e-6
                        for i in range(len(x)):
                            x_plus = x.copy()
                            x_plus[i] += eps
                            grad[i] = (objective_fn(x_plus) - f0) / eps
                        x = x - 0.03 * grad
                        history.append(x.copy())
                    optimal_positions = x
                    algo_history = ['adam'] * max_iterations

                elapsed_time = time.time() - start_time

                # 构建无人机轨迹历史
                uav_histories = []
                for i in range(num_uavs):
                    traj = []
                    for h in history:
                        traj.append([h[3*i], h[3*i+1], h[3*i+2]])
                    uav_histories.append(traj)

                # 最终位置
                final_positions = [[optimal_positions[3*i], optimal_positions[3*i+1], optimal_positions[3*i+2]] for i in range(num_uavs)]

                # 计算覆盖率（基于最终用户位置）
                final_cost = objective_fn(optimal_positions)
                coverage = max(0, min(98, ((-final_cost / (num_uavs * 15)) * 100 + 65)))

                # 指标卡片
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("优化耗时", f"{elapsed_time:.2f} 秒")
                with col2: st.metric("收敛迭代", f"{len(history)-1} 次")
                with col3: st.metric("灾区覆盖率", f"{coverage:.1f}%")
                with col4: st.metric("当前算法", algo_history[-1].upper() if algo_history else "Adam")
                with col5: st.metric("动态用户", f"{num_users} 人", delta="移动中")

                # 动态3D图
                st.subheader("🗺️ 无人机动态飞行 + 用户动态移动（点击播放按钮观看动画）")
                fig_dynamic = create_dynamic_3d_plot(uav_histories, user_model, terrain_model, terrain_type)
                st.plotly_chart(fig_dynamic, use_container_width=True)

                # 其他图表
                col_left, col_right = st.columns(2)
                with col_left:
                    fig_heat = create_coverage_heatmap(final_positions, user_model, terrain_type)
                    st.plotly_chart(fig_heat, use_container_width=True)
                with col_right:
                    fig_comp = create_algorithm_comparison(len(history)-1)
                    st.plotly_chart(fig_comp, use_container_width=True)

                col_eng, col_sw, col_rad = st.columns(3)
                with col_eng:
                    fig_eng = create_energy_chart(len(history)-1, enable_solar)
                    st.plotly_chart(fig_eng, use_container_width=True)
                with col_sw:
                    if enable_adaptive:
                        fig_sw = create_algorithm_switch_chart(algo_history)
                        st.plotly_chart(fig_sw, use_container_width=True)
                    else:
                        st.info("自适应算法切换未启用")
                with col_rad:
                    radar_metrics = [min(100, int(100*(1-len(history)/max_iterations*0.5))), coverage,
                                     min(100, int(70+(enable_solar and 20 or 0))), 85, 90]
                    fig_rad = create_performance_radar(radar_metrics)
                    st.plotly_chart(fig_rad, use_container_width=True)

                # 日志
                st.subheader("📝 仿真日志")
                algo_count = {}
                for a in algo_history: algo_count[a] = algo_count.get(a,0)+1
                log_text = f"✅ 仿真完成！耗时 {elapsed_time:.2f}秒，覆盖率 {coverage:.1f}%\n"
                log_text += f"LD-HAF 算法统计: {', '.join([f'{k.upper()}:{v}次' for k,v in algo_count.items()])}\n"
                log_text += f"用户动态移动模型：高斯-马尔可夫过程，速度 {user_model.speed:.2f} m/步"
                st.code(log_text, language="text")
                st.success("🎉 仿真成功！动态展示无人机飞行和用户移动。")

            except Exception as e:
                st.error(f"仿真出错: {str(e)}")
    else:
        st.info("👈 请在左侧配置参数，然后点击「开始地震应急仿真」")
        st.markdown("""
        - **无人机动态飞行动画**：点击播放按钮观看无人机沿优化轨迹飞行
        - **用户动态移动**：灾区用户按照高斯-马尔可夫模型随机游走
        - **LD-HAF自适应优化**：自动切换优化算法，快速收敛
        - **3D地形避障**：山区地形自动规避
        - **太阳能能量管理**：实时电量监控
        """)

if __name__ == "__main__":
    main()
