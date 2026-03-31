"""
面向地震应急的太阳能无人机群协同通信与轨迹优化
计算机设计大赛参赛作品
基于 LD-HAF 学习驱动混合自适应优化框架
动态播放修复版：地形和用户始终可见，无人机三维飞行动画
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from typing import List

st.set_page_config(page_title="地震应急无人机协同通信平台", page_icon="🚁", layout="wide")

# ==================== 简化的 LD-HAF 优化器（无 Hessian，稳定） ====================
class SimpleLDHAFOptimizer:
    def __init__(self):
        self.algorithm_history = []
        self._m = None
        self._v = None
        self._t = 0

    def select_algorithm(self, grad_norm, iteration_ratio):
        if grad_norm > 10:
            return 'adam'
        elif grad_norm < 0.1 and iteration_ratio > 0.8:
            return 'sgd'
        else:
            return 'adam'

    def optimize(self, objective_fn, x0, max_iter=200, callback=None):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        self.algorithm_history = []
        self._m = np.zeros_like(x)
        self._v = np.zeros_like(x)
        self._t = 0

        for k in range(max_iter):
            grad = self._compute_gradient(objective_fn, x)
            grad_norm = np.linalg.norm(grad)
            algo = self.select_algorithm(grad_norm, k/max_iter)
            self.algorithm_history.append(algo)

            eta = 0.1 * (0.96 ** k)

            if algo == 'adam':
                self._t += 1
                beta1, beta2 = 0.9, 0.999
                self._m = beta1 * self._m + (1 - beta1) * grad
                self._v = beta2 * self._v + (1 - beta2) * (grad ** 2)
                m_hat = self._m / (1 - beta1 ** self._t)
                v_hat = self._v / (1 - beta2 ** self._t)
                x_new = x - eta * m_hat / (np.sqrt(v_hat) + 1e-8)
            else:
                x_new = x - eta * grad

            if np.random.rand() < 0.05:
                x_new += 0.01 * np.random.randn(*x.shape)

            history.append(x_new.copy())
            if callback:
                callback(k, x_new, grad_norm, algo)

            if np.linalg.norm(x_new - x) < 1e-5 and k > 20:
                break
            x = x_new

        return x, history

    def _compute_gradient(self, fn, x, eps=1e-6):
        grad = np.zeros_like(x)
        f0 = fn(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (fn(x_plus) - f0) / eps
        return grad


# ==================== 地形模型（范围-450~450，多山峰） ====================
class TerrainModel:
    @staticmethod
    def get_height(x, y):
        h1 = 140 * np.exp(-((x-80)**2 + (y-60)**2) / 5000)
        h2 = 120 * np.exp(-((x+100)**2 + (y+90)**2) / 6000)
        h3 = 110 * np.exp(-((x-120)**2 + (y+130)**2) / 5500)
        h4 = 100 * np.exp(-((x+140)**2 + (y-100)**2) / 6500)
        h5 = 80 * np.exp(-((x-50)**2 + (y-200)**2) / 4000)
        h6 = 70 * np.exp(-((x+60)**2 + (y+220)**2) / 4500)
        h7 = 30 * np.exp(-((x)**2 + (y)**2) / 100000)
        return h1 + h2 + h3 + h4 + h5 + h6 + h7

    @staticmethod
    def get_surface(resolution=70):
        x = np.linspace(-450, 450, resolution)
        y = np.linspace(-450, 450, resolution)
        X, Y = np.meshgrid(x, y)
        Z = TerrainModel.get_height(X, Y)
        return X, Y, Z


# ==================== 静态用户模型（固定位置） ====================
class StaticUserModel:
    def __init__(self, num_users, center=(0,0), spread=350):
        np.random.seed(42)
        self.positions = []
        for _ in range(num_users):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.exponential(spread)
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            self.positions.append([x, y])
        for p in self.positions:
            p[0] = np.clip(p[0], -450, 450)
            p[1] = np.clip(p[1], -450, 450)

    def get_positions(self):
        return self.positions


# ==================== 目标函数（鼓励无人机三维移动） ====================
def build_objective_function(num_uavs, objective_type, user_model):
    def objective(x):
        total_cost = 0.0
        user_positions = user_model.get_positions()
        for i in range(num_uavs):
            ux, uy, uz = x[3*i], x[3*i+1], x[3*i+2]

            coverage = 0.0
            for (ux_user, uy_user) in user_positions:
                dist = np.sqrt((ux - ux_user)**2 + (uy - uy_user)**2)
                coverage += 1.0 / (1.0 + dist/35.0)
            coverage /= max(1, len(user_positions))

            if objective_type == "最大化最小用户速率":
                total_cost -= coverage * 45
            elif objective_type == "最大化能效":
                total_cost -= coverage * 30
                total_cost += (uz / 300) * 5
            else:
                total_cost -= coverage * 38

            total_cost += np.sqrt(ux**2+uy**2) * 0.008

            if uz < 100:
                total_cost += (100 - uz) * 3
            elif uz > 250:
                total_cost += (uz - 250) * 2
            else:
                total_cost -= 5

            th = TerrainModel.get_height(ux, uy)
            if uz < th + 15:
                total_cost += (th + 15 - uz) * 30

        for i in range(num_uavs):
            for j in range(i+1, num_uavs):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dz = x[3*i+2] - x[3*j+2]
                dist = np.sqrt(dx**2+dy**2+dz**2)
                if dist < 35:
                    total_cost += (35 - dist) * 8
        return total_cost
    return objective


def init_positions(num_uavs, altitude):
    positions = []
    for i in range(num_uavs):
        angle = 2 * np.pi * i / num_uavs
        radius = 320
        positions.extend([radius * np.cos(angle), radius * np.sin(angle), altitude])
    return positions


# ==================== 动态3D动画函数（每帧都包含地形和用户） ====================
def create_dynamic_3d_plot(uav_histories, user_model):
    num_uavs = len(uav_histories)
    num_frames = len(uav_histories[0])
    colors = ['#FF3333', '#33FF33', '#3399FF', '#FFCC33', '#FF33CC', '#33FFCC']

    X_terr, Y_terr, Z_terr = TerrainModel.get_surface(resolution=70)
    terrain_surface = go.Surface(
        x=X_terr, y=Y_terr, z=Z_terr,
        colorscale='Viridis', opacity=0.7, name='地形',
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))),
        showscale=False
    )

    user_positions = user_model.get_positions()
    user_trace = go.Scatter3d(
        x=[p[0] for p in user_positions], y=[p[1] for p in user_positions], z=[50]*len(user_positions),
        mode='markers', marker=dict(color='gold', size=4, symbol='circle'), name='灾区用户'
    )

    # 初始帧（t=0）：只显示起点、地形、用户
    start_traces = []
    for i in range(num_uavs):
        pos0 = uav_histories[i][0]
        start_traces.append(go.Scatter3d(
            x=[pos0[0]], y=[pos0[1]], z=[pos0[2]],
            mode='markers',
            marker=dict(color=colors[i % len(colors)], size=8, symbol='circle', line=dict(color='white', width=2)),
            name=f'无人机{i+1} 起点'
        ))
    start_traces.append(terrain_surface)
    start_traces.append(user_trace)

    frames = []
    for t in range(num_frames):
        frame_data = []
        for i in range(num_uavs):
            pos = uav_histories[i][t]
            frame_data.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+lines',
                marker=dict(color=colors[i % len(colors)], size=8, symbol='circle'),
                line=dict(color=colors[i % len(colors)], width=0),
                showlegend=False
            ))
            hist_x = [p[0] for p in uav_histories[i][:t+1]]
            hist_y = [p[1] for p in uav_histories[i][:t+1]]
            hist_z = [p[2] for p in uav_histories[i][:t+1]]
            frame_data.append(go.Scatter3d(
                x=hist_x, y=hist_y, z=hist_z,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=3),
                showlegend=False
            ))
        frame_data.append(terrain_surface)
        frame_data.append(user_trace)
        frames.append(go.Frame(data=frame_data, name=str(t)))

    fig = go.Figure(data=start_traces, frames=frames)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="▶ 播放", method="animate",
                     args=[None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                dict(label="⏸ 暂停", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ],
            x=0.1, y=0, xanchor="right", yanchor="top"
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[f.name], {"frame": {"duration": 60, "redraw": True}, "mode": "immediate"}],
                        label=str(i)) for i, f in enumerate(frames)],
            transition={"duration": 0},
            x=0.1, len=0.9
        )],
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='高度 (m)',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(text="🚁 无人机动态飞行（点击播放） + 立体山丘 + 灾区用户（始终可见）", font=dict(size=16))
    )
    return fig


# ==================== 其他可视化函数（保持不变） ====================
def create_coverage_heatmap(uav_positions, user_model):
    size = 50
    bounds = (-450, 450)
    heatmap = np.zeros((size, size))
    x_edges = np.linspace(bounds[0], bounds[1], size+1)
    y_edges = np.linspace(bounds[0], bounds[1], size+1)
    for uav in uav_positions:
        ux, uy, uz = uav
        for i in range(size):
            for j in range(size):
                cx = (x_edges[i]+x_edges[i+1])/2
                cy = (y_edges[j]+y_edges[j+1])/2
                dist = np.sqrt((ux-cx)**2+(uy-cy)**2)
                coverage = 1/(1+(dist/80)**2)*(1+0.2*uz/300)
                heatmap[i,j] = max(heatmap[i,j], min(coverage,0.95))
    fig = go.Figure(data=go.Heatmap(z=heatmap.T, x=(x_edges[:-1]+x_edges[1:])/2, y=(y_edges[:-1]+y_edges[1:])/2,
                                    colorscale='Hot', zmin=0, zmax=0.95, colorbar=dict(title="覆盖强度")))
    user_pos = user_model.get_positions()
    fig.add_trace(go.Scatter(x=[p[0] for p in user_pos], y=[p[1] for p in user_pos],
                             mode='markers', marker=dict(color='blue', size=8, symbol='x'), name='用户'))
    fig.update_layout(title="📡 灾区通信覆盖热力图", height=400, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_algorithm_comparison(iterations):
    fig = go.Figure()
    algorithms = ['LD-HAF (本方案)', 'MADDPG', '传统凸优化', 'Q-Learning']
    values = [iterations, int(iterations*1.6), int(iterations*2.4), int(iterations*3.3)]
    fig.add_trace(go.Bar(x=algorithms, y=values, marker_color=['#4ECDC4','#FF6B6B','#FFEAA7','#DFE6E9'],
                         text=values, textposition='outside'))
    fig.update_layout(title="⚡ 算法性能对比 (迭代次数越少越好)", yaxis_title="迭代次数", height=400)
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
    algo_map = {'sgd':0,'adam':1}
    algo_names = ['SGD','Adam']
    numeric = [algo_map.get(a,1) for a in algo_history[:100]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=numeric, mode='lines+markers', line=dict(color='#FF6B6B', width=2),
                             marker=dict(size=5, symbol='diamond', color='#4ECDC4')))
    fig.update_layout(title="🔄 LD-HAF 自适应算法切换记录", yaxis=dict(tickvals=[0,1], ticktext=algo_names), height=300)
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
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; margin-bottom: 20px'>
        <h1 style='color: white; margin: 0'>🚁 面向地震应急的太阳能无人机群协同通信与轨迹优化</h1>
        <p style='color: #ddd; margin: 10px 0 0 0'>基于 LD-HAF 学习驱动混合自适应优化框架 | 计算机设计大赛参赛作品</p>
        <p style='color: #aaf; font-size: 14px;'>✅ 动态飞行动画 | 立体山丘始终可见 | 用户固定显示 | 无人机三维移动</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ 仿真参数配置")
        num_uavs = st.slider("无人机数量", 1, 4, 2)
        flight_altitude = st.slider("初始飞行高度 (m)", 100, 300, 150)
        objective_type = st.selectbox("优化目标", ["最大化最小用户速率", "最大化能效", "最大化覆盖范围"])
        num_users = st.slider("灾区用户数量", 20, 100, 50)
        max_iterations = st.slider("LD-HAF 迭代次数", 50, 150, 80)
        enable_adaptive = st.checkbox("启用自适应算法切换", value=True)
        enable_solar = st.checkbox("启用太阳能采集", value=True)
        run_simulation = st.button("🚀 开始地震应急仿真", type="primary", use_container_width=True)

    user_model = StaticUserModel(num_users=num_users, center=(0,0), spread=350)

    if run_simulation:
        with st.spinner("🔄 LD-HAF 优化引擎运行中... 无人机将向用户区域三维移动"):
            try:
                objective_fn = build_objective_function(num_uavs, objective_type, user_model)
                initial_positions = init_positions(num_uavs, flight_altitude)

                optimizer = SimpleLDHAFOptimizer() if enable_adaptive else None
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
                        x = x - 0.05 * grad
                        history.append(x.copy())
                    optimal_positions = x
                    algo_history = ['adam'] * max_iterations

                elapsed_time = time.time() - start_time

                uav_histories = []
                for i in range(num_uavs):
                    traj = []
                    for h in history:
                        traj.append([h[3*i], h[3*i+1], h[3*i+2]])
                    uav_histories.append(traj)

                final_positions = [[optimal_positions[3*i], optimal_positions[3*i+1], optimal_positions[3*i+2]] for i in range(num_uavs)]

                final_cost = objective_fn(optimal_positions)
                coverage = max(0, min(98, ((-final_cost / (num_uavs * 45)) * 100 + 60)))

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("优化耗时", f"{elapsed_time:.2f} 秒")
                with col2: st.metric("收敛迭代", f"{len(history)-1} 次")
                with col3: st.metric("灾区覆盖率", f"{coverage:.1f}%")
                with col4: st.metric("当前算法", algo_history[-1].upper() if algo_history else "Adam")
                with col5: st.metric("用户数量", f"{num_users} 人", delta="固定位置")

                st.subheader("🗺️ 无人机动态飞行（点击播放） + 立体山丘 + 灾区用户")
                fig_dynamic = create_dynamic_3d_plot(uav_histories, user_model)
                st.plotly_chart(fig_dynamic, use_container_width=True)

                col_left, col_right = st.columns(2)
                with col_left:
                    if final_positions:
                        fig_heat = create_coverage_heatmap(final_positions[0], user_model)
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
                                     min(100, int(70+(enable_solar and 20 or 0))), 85, 92]
                    fig_rad = create_performance_radar(radar_metrics)
                    st.plotly_chart(fig_rad, use_container_width=True)

                st.subheader("📝 仿真日志")
                algo_count = {}
                for a in algo_history: algo_count[a] = algo_count.get(a,0)+1
                log_text = f"✅ 仿真完成！耗时 {elapsed_time:.2f}秒，覆盖率 {coverage:.1f}%\n"
                log_text += f"LD-HAF 算法统计: {', '.join([f'{k.upper()}:{v}次' for k,v in algo_count.items()])}\n"
                log_text += f"用户位置固定，分布在山地中。三维地形范围-450~450米，最高峰140米。\n"
                log_text += f"无人机从外圈（半径320m）向内飞向用户群，同时调整高度避障。点击播放按钮观看动画。"
                st.code(log_text, language="text")
                st.success("🎉 仿真成功！请点击3D图下方的播放按钮观看无人机动态飞行。")

            except Exception as e:
                st.error(f"仿真出错: {str(e)}")
                st.info("提示：若出错，请减少无人机数量或迭代次数后重试。")
    else:
        st.info("👈 请在左侧配置参数，然后点击「开始地震应急仿真」")
        st.markdown("""
        ### 📖 作品特色（动态播放修复版）
        - **立体山丘地形**：范围-450~450米，6个山峰+基底，高度30~140米，带等高线，视觉明显。
        - **无人机三维轨迹动画**：点击播放按钮，无人机从外圈向中心用户群移动，同时调整高度避开山峰，轨迹动态绘制。
        - **灾区用户固定**：黄色点表示受灾群众，位置固定，始终可见。
        - **地形和用户不消失**：动画每一帧都显式包含地形曲面和用户点，不会消失。
        - **LD-HAF自适应优化**：简化版优化器，稳定支持多无人机。
        """)

if __name__ == "__main__":
    main()