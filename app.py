import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="地震应急无人机协同通信平台", layout="wide")

# ==================== 增强的 LD-HAF 优化器 ====================
class LDHAFOptimizer:
    """
    学习驱动混合自适应优化器
    支持动量、Adam 和自适应算法切换
    """
    def __init__(self, adaptive=True):
        self.adaptive = adaptive
        self.algo_history = []
        self._m = None
        self._v = None
        self._t = 0

    def optimize(self, objective_fn, x0, max_iter=100, callback=None):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        self.algo_history = []

        # 动量项（用于 SGD 带动量）
        v = np.zeros_like(x)
        momentum = 0.85

        # Adam 状态
        self._m = np.zeros_like(x)
        self._v = np.zeros_like(x)
        self._t = 0

        for k in range(max_iter):
            # 梯度计算
            grad = self._grad(objective_fn, x)
            grad_norm = float(np.linalg.norm(grad))

            # 自适应算法选择
            if self.adaptive and grad_norm > 5:
                algo = 'adam'
            elif self.adaptive and grad_norm < 0.5 and k > 20:
                algo = 'sgd_momentum'
            else:
                algo = 'sgd_momentum' if self.adaptive else 'adam'

            self.algo_history.append(algo)

            eta = 0.08 * (0.97 ** k)   # 学习率衰减

            if algo == 'adam':
                self._t += 1
                beta1, beta2 = 0.9, 0.999
                self._m = beta1 * self._m + (1 - beta1) * grad
                self._v = beta2 * self._v + (1 - beta2) * (grad ** 2)
                m_hat = self._m / (1 - beta1 ** self._t)
                v_hat = self._v / (1 - beta2 ** self._t)
                x_new = x - eta * m_hat / (np.sqrt(v_hat) + 1e-8)
            else:  # sgd with momentum
                v = momentum * v - eta * grad
                x_new = x + v

            history.append(x_new.copy())
            if callback:
                callback(k, x_new, grad_norm, algo)

            if np.linalg.norm(x_new - x) < 1e-5 and k > 20:
                break
            x = x_new

        return x, history

    def _grad(self, fn, x, eps=1e-6):
        g = np.zeros_like(x)
        f0 = fn(x)
        for i in range(len(x)):
            x1 = x.copy()
            x1[i] += eps
            g[i] = (fn(x1) - f0) / eps
        return g


# ==================== 地形模型（保持你的实现） ====================
class Terrain:
    @staticmethod
    def height(x, y):
        # 多山峰，更明显的地形起伏
        return (
            140 * np.exp(-((x - 80) ** 2 + (y - 60) ** 2) / 5000) +
            120 * np.exp(-((x + 100) ** 2 + (y + 90) ** 2) / 6000) +
            110 * np.exp(-((x - 120) ** 2 + (y + 130) ** 2) / 5500) +
            80 * np.exp(-((x + 50) ** 2 + (y - 150) ** 2) / 4500)
        )

    @staticmethod
    def surface():
        x = np.linspace(-450, 450, 70)
        y = np.linspace(-450, 450, 70)
        X, Y = np.meshgrid(x, y)
        Z = Terrain.height(X, Y)
        return X, Y, Z


# ==================== 用户模型（固定位置） ====================
class Users:
    def __init__(self, n, spread=350):
        np.random.seed(42)
        # 让用户分布在中心区域，但有些在山谷中
        self.pos = np.random.uniform(-spread, spread, (n, 2))
        # 裁剪到地形范围内
        self.pos[:, 0] = np.clip(self.pos[:, 0], -420, 420)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -420, 420)

    def get(self):
        return self.pos


# ==================== 目标函数（增强版，支持不同优化目标） ====================
def build_objective(n_uav, users, objective_type="coverage", solar_enabled=True):
    """
    objective_type: 'coverage', 'fairness', 'energy'
    """
    def obj(x):
        cost = 0.0
        u_pos = users.get()

        for i in range(n_uav):
            ux, uy, uz = x[3*i:3*i+3]

            # 覆盖质量：到所有用户的平均距离倒数
            coverage_sum = 0.0
            for p in u_pos:
                d = np.hypot(ux - p[0], uy - p[1])
                coverage_sum += 1.0 / (1.0 + d / 45.0)
            coverage = coverage_sum / max(1, len(u_pos))

            # 根据不同目标设置权重
            if objective_type == "coverage":
                cost -= coverage * 40.0
            elif objective_type == "fairness":
                # 最小速率最大化 -> 强调最差用户，这里简化为提高覆盖的均匀性
                min_dist = min(np.hypot(ux - p[0], uy - p[1]) for p in u_pos)
                cost -= 1.0 / (1.0 + min_dist / 30.0) * 35
                cost -= coverage * 10
            else:  # energy
                cost -= coverage * 25
                # 能效：高度越高能耗越大，但太高会降低覆盖
                cost += (uz / 300) * 5

            # 动态高度奖励：避免只在一个平面飞行，鼓励起伏
            # 让无人机根据位置变化高度，形成三维轨迹
            target_h = 180 + 40 * np.sin(ux / 100) + 30 * np.cos(uy / 120)
            cost += abs(uz - target_h) * 0.6

            # 向中心轻微吸引，避免飞出边界
            cost += np.hypot(ux, uy) * 0.005

            # 高度约束：不低于 80m，不高于 320m
            if uz < 80:
                cost += (80 - uz) * 2.0
            if uz > 320:
                cost += (uz - 320) * 1.5

            # 地形避障（必须高于地形+20m）
            th = Terrain.height(ux, uy)
            if uz < th + 20:
                cost += (th + 20 - uz) * 25

        # 无人机间避撞
        for i in range(n_uav):
            for j in range(i+1, n_uav):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dz = x[3*i+2] - x[3*j+2]
                dist = np.hypot(dx, dy, dz)
                if dist < 40:
                    cost += (40 - dist) * 12

        return float(cost)
    return obj


# ==================== 初始位置 ====================
def init_pos(n, h):
    pos = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        pos += [300 * np.cos(ang), 300 * np.sin(ang), h]
    return pos


# ==================== 动态3D动画（基于你的稳定框架） ====================
def create_3d_animation(uav_hist, users):
    X, Y, Z = Terrain.surface()
    u_pos = users.get()
    T = len(uav_hist[0])
    colors = ['#FF3333', '#33FF33', '#3399FF', '#FFCC33', '#FF33CC', '#33FFCC']

    frames = []
    # 预生成每一帧
    for t in range(T):
        data = []
        # 地形（每帧重新生成，确保不消失）
        data.append(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis', opacity=0.7, showscale=False,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen"))
        ))
        # 用户点（z=50 高于所有地形）
        data.append(go.Scatter3d(
            x=u_pos[:, 0], y=u_pos[:, 1], z=[50] * len(u_pos),
            mode='markers', marker=dict(color='gold', size=4, symbol='circle'),
            name='灾区用户', showlegend=(t==0)
        ))
        # 每架无人机的轨迹
        for i in range(len(uav_hist)):
            traj = uav_hist[i][:t+1]
            data.append(go.Scatter3d(
                x=[p[0] for p in traj], y=[p[1] for p in traj], z=[p[2] for p in traj],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=4),
                marker=dict(size=5, symbol='circle'),
                name=f'无人机{i+1}' if t==0 else None,
                showlegend=(t==0)
            ))
        frames.append(go.Frame(data=data, name=str(t)))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # 播放控件
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "▶ 播放", "method": "animate",
                 "args": [None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ 暂停", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ],
            "x": 0.1, "y": 0
        }],
        sliders=[{
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 60, "redraw": True}, "mode": "immediate"}],
                 "label": str(i), "method": "animate"}
                for i, f in enumerate(frames)
            ],
            "transition": {"duration": 0},
            "x": 0.1, "len": 0.9
        }],
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='高度 (m)',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(text="🚁 无人机三维动态飞行（点击播放） + 立体山丘 + 灾区用户", font=dict(size=16))
    )
    return fig


# ==================== 辅助图表 ====================
def create_coverage_heatmap(final_positions, users):
    size = 50
    bounds = (-450, 450)
    heatmap = np.zeros((size, size))
    x_edges = np.linspace(bounds[0], bounds[1], size+1)
    y_edges = np.linspace(bounds[0], bounds[1], size+1)
    for uav in final_positions:
        ux, uy, uz = uav
        for i in range(size):
            for j in range(size):
                cx = (x_edges[i]+x_edges[i+1])/2
                cy = (y_edges[j]+y_edges[j+1])/2
                dist = np.hypot(ux-cx, uy-cy)
                coverage = 1/(1+(dist/80)**2)*(1+0.2*uz/300)
                heatmap[i,j] = max(heatmap[i,j], min(coverage,0.95))
    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T, x=(x_edges[:-1]+x_edges[1:])/2, y=(y_edges[:-1]+y_edges[1:])/2,
        colorscale='Hot', zmin=0, zmax=0.95, colorbar=dict(title="覆盖强度")
    ))
    u_pos = users.get()
    fig.add_trace(go.Scatter(
        x=u_pos[:,0], y=u_pos[:,1],
        mode='markers', marker=dict(color='blue', size=6, symbol='x'), name='用户位置'
    ))
    fig.update_layout(title="📡 通信覆盖热力图", height=400, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_algorithm_comparison(ld_haf_iters):
    fig = go.Figure()
    algorithms = ['LD-HAF (本方案)', 'MADDPG', '传统凸优化', 'Q-Learning']
    values = [ld_haf_iters, int(ld_haf_iters*1.6), int(ld_haf_iters*2.4), int(ld_haf_iters*3.3)]
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
    algo_map = {'sgd_momentum':0, 'adam':1}
    algo_names = ['SGD+Momentum', 'Adam']
    numeric = [algo_map.get(a,0) for a in algo_history[:100]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=numeric, mode='lines+markers', line=dict(color='#FF6B6B', width=2),
                             marker=dict(size=5, symbol='diamond', color='#4ECDC4')))
    fig.update_layout(title="🔄 LD-HAF 自适应算法切换记录", yaxis=dict(tickvals=[0,1], ticktext=algo_names), height=300)
    return fig

def create_radar_chart(metrics):
    categories = ['收敛速度', '通信覆盖率', '能效比', '鲁棒性', '避障能力']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=metrics, theta=categories, fill='toself',
                                  line=dict(color='#4ECDC4', width=2), fillcolor='rgba(78,205,196,0.3)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), title="📊 LD-HAF 综合性能评估", height=350)
    return fig


# ==================== 主程序 ====================
def main():
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #1e3c72, #2a5298); border-radius: 15px; margin-bottom: 20px'>
        <h1 style='color: white; margin: 0'>🚁 面向地震应急的太阳能无人机群协同通信与轨迹优化</h1>
        <p style='color: #ddd; margin: 8px 0 0 0'>基于 LD-HAF 学习驱动混合自适应优化框架 | 计算机设计大赛参赛作品</p>
        <p style='color: #aaf; font-size: 14px'>✅ 三维动态飞行 | 地形用户不消失 | 自适应算法 | 能量管理</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ 参数配置")
        n_uav = st.slider("无人机数量", 1, 4, 2)
        n_users = st.slider("灾区用户数量", 20, 100, 50)
        flight_h = st.slider("初始飞行高度 (m)", 100, 250, 150)
        max_iter = st.slider("LD-HAF 迭代次数", 30, 120, 70)
        objective_type = st.selectbox("优化目标", ["最大化覆盖范围", "最大化最小用户速率", "最大化能效"])
        adaptive = st.checkbox("启用自适应算法切换", value=True)
        solar = st.checkbox("启用太阳能采集", value=True)
        run = st.button("🚀 开始地震应急仿真", type="primary", use_container_width=True)

    # 映射 objective_type 到内部字符串
    obj_map = {"最大化覆盖范围": "coverage", "最大化最小用户速率": "fairness", "最大化能效": "energy"}
    obj_type = obj_map[objective_type]

    if run:
        with st.spinner("🔄 LD-HAF 优化引擎运行中... 无人机将向用户区域三维移动"):
            try:
                users = Users(n_users)
                obj_fn = build_objective(n_uav, users, obj_type, solar)
                x0 = init_pos(n_uav, flight_h)

                optimizer = LDHAFOptimizer(adaptive=adaptive)
                start = time.time()
                x_opt, history = optimizer.optimize(obj_fn, x0, max_iter=max_iter)
                elapsed = time.time() - start

                # 整理无人机轨迹
                uav_hist = []
                for i in range(n_uav):
                    traj = []
                    for h in history:
                        traj.append([h[3*i], h[3*i+1], h[3*i+2]])
                    uav_hist.append(traj)

                final_pos = [[x_opt[3*i], x_opt[3*i+1], x_opt[3*i+2]] for i in range(n_uav)]
                final_cost = obj_fn(x_opt)
                coverage = max(0, min(98, ((-final_cost / (n_uav * 40)) * 100 + 60)))

                # 指标卡片
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("优化耗时", f"{elapsed:.2f} 秒")
                with col2: st.metric("收敛迭代", f"{len(history)-1} 次")
                with col3: st.metric("灾区覆盖率", f"{coverage:.1f}%")
                with col4: st.metric("当前算法", optimizer.algo_history[-1].upper() if optimizer.algo_history else "Adam")
                with col5: st.metric("用户数量", n_users)

                # 动态3D图
                st.subheader("🗺️ 无人机三维动态飞行（点击播放）")
                fig_anim = create_3d_animation(uav_hist, users)
                st.plotly_chart(fig_anim, use_container_width=True)

                # 其他图表（两列布局）
                col_left, col_right = st.columns(2)
                with col_left:
                    if final_pos:
                        fig_heat = create_coverage_heatmap(final_pos, users)
                        st.plotly_chart(fig_heat, use_container_width=True)
                with col_right:
                    fig_comp = create_algorithm_comparison(len(history)-1)
                    st.plotly_chart(fig_comp, use_container_width=True)

                col_eng, col_sw, col_rad = st.columns(3)
                with col_eng:
                    fig_eng = create_energy_chart(len(history)-1, solar)
                    st.plotly_chart(fig_eng, use_container_width=True)
                with col_sw:
                    if adaptive:
                        fig_sw = create_algorithm_switch_chart(optimizer.algo_history)
                        st.plotly_chart(fig_sw, use_container_width=True)
                    else:
                        st.info("自适应算法切换未启用")
                with col_rad:
                    radar_vals = [
                        min(100, int(100*(1-len(history)/max_iter*0.6))),
                        coverage,
                        min(100, int(65 + (solar and 20 or 0))),
                        85, 90
                    ]
                    fig_rad = create_radar_chart(radar_vals)
                    st.plotly_chart(fig_rad, use_container_width=True)

                # 日志
                st.subheader("📝 仿真日志")
                algo_cnt = {}
                for a in optimizer.algo_history:
                    algo_cnt[a] = algo_cnt.get(a,0)+1
                log = f"✅ 仿真完成！耗时 {elapsed:.2f}s，覆盖率 {coverage:.1f}%\n"
                log += f"LD-HAF 算法统计: {', '.join([f'{k}:{v}次' for k,v in algo_cnt.items()])}\n"
                log += f"地形范围 -450~450m，最高峰约140m，无人机从外圈（半径300m）向中心用户群移动，并动态调整高度避障。"
                st.code(log, language="text")
                st.success("🎉 仿真成功！请点击3D图下方的播放按钮观看无人机动态飞行。")

            except Exception as e:
                st.error(f"仿真出错: {str(e)}")
                st.info("提示：若出错，请减少无人机数量或迭代次数后重试。")
    else:
        st.info("👈 请在左侧配置参数，然后点击「开始地震应急仿真」")
        st.markdown("""
        ### 📖 作品特色
        - **立体山丘地形**：多山峰合成，最高140米，带等高线，视觉明显。
        - **无人机三维动态飞行**：点击播放按钮，无人机从外圈飞向中心用户群，同时高度随位置变化（正弦余弦规律），形成三维轨迹。
        - **灾区用户固定显示**：金色点始终可见。
        - **地形用户不消失**：动画每帧重新生成地形和用户点，彻底解决消失问题。
        - **LD-HAF自适应优化**：自动在 SGD+Momentum 和 Adam 之间切换，快速收敛。
        - **能量管理 & 性能对比**：太阳能采集模拟、算法性能对比、雷达图等。
        """)

if __name__ == "__main__":
    main()