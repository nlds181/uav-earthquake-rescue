import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="无人机协同通信", layout="wide")

# ==================== 优化器（加入动量） ====================
class SimpleLDHAFOptimizer:
    def optimize(self, objective_fn, x0, max_iter=100):
        x = np.array(x0, dtype=float)
        history = [x.copy()]

        v = np.zeros_like(x)  # 动量
        momentum = 0.85

        for k in range(max_iter):
            grad = self._grad(objective_fn, x)

            eta = 0.08 * (0.97 ** k)

            # 🚀 动量更新（关键）
            v = momentum * v - eta * grad
            x = x + v

            history.append(x.copy())

        return x, history

    def _grad(self, fn, x, eps=1e-6):
        g = np.zeros_like(x)
        f0 = fn(x)
        for i in range(len(x)):
            x1 = x.copy()
            x1[i] += eps
            g[i] = (fn(x1) - f0) / eps
        return g


# ==================== 地形 ====================
class Terrain:
    @staticmethod
    def height(x, y):
        return (
            140*np.exp(-((x-80)**2+(y-60)**2)/5000) +
            120*np.exp(-((x+100)**2+(y+90)**2)/6000) +
            110*np.exp(-((x-120)**2+(y+130)**2)/5500)
        )

    @staticmethod
    def surface():
        x = np.linspace(-450, 450, 70)
        y = np.linspace(-450, 450, 70)
        X, Y = np.meshgrid(x, y)
        Z = Terrain.height(X, Y)
        return X, Y, Z


# ==================== 用户 ====================
class Users:
    def __init__(self, n):
        np.random.seed(42)
        self.pos = np.random.uniform(-300, 300, (n, 2))

    def get(self):
        return self.pos


# ==================== 目标函数（核心升级） ====================
def build_objective(n_uav, users):
    def obj(x):
        cost = 0
        u_pos = users.get()

        for i in range(n_uav):
            ux, uy, uz = x[3*i:3*i+3]

            # 覆盖
            for p in u_pos:
                d = np.sqrt((ux-p[0])**2 + (uy-p[1])**2)
                cost -= 1/(1+d/50)

            # 🚀 三维动态高度（关键）
            optimal_h = 180 + 40*np.sin(ux/100) + 30*np.cos(uy/120)
            cost += abs(uz - optimal_h) * 0.8

            # 避障
            th = Terrain.height(ux, uy)
            if uz < th + 20:
                cost += (th + 20 - uz) * 20

        return cost
    return obj


# ==================== 初始位置 ====================
def init_pos(n, h):
    pos = []
    for i in range(n):
        ang = 2*np.pi*i/n
        pos += [300*np.cos(ang), 300*np.sin(ang), h]
    return pos


# ==================== 动态3D（完全修复版） ====================
def create_3d(uav_hist, users):
    X, Y, Z = Terrain.surface()
    u_pos = users.get()

    frames = []
    T = len(uav_hist[0])
    colors = ['red', 'blue', 'green', 'orange']

    for t in range(T):
        data = []

        # 🚀 每帧重新生成地形（关键修复）
        data.append(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.7,
            showscale=False
        ))

        # 用户
        data.append(go.Scatter3d(
            x=u_pos[:,0], y=u_pos[:,1], z=[50]*len(u_pos),
            mode='markers',
            marker=dict(color='gold', size=4),
            showlegend=False
        ))

        # 无人机轨迹
        for i in range(len(uav_hist)):
            traj = uav_hist[i][:t+1]

            data.append(go.Scatter3d(
                x=[p[0] for p in traj],
                y=[p[1] for p in traj],
                z=[p[2] for p in traj],
                mode='lines+markers',
                line=dict(width=4),
                marker=dict(size=4),
                showlegend=False
            ))

        frames.append(go.Frame(data=data, name=str(t)))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ 播放", "method": "animate",
                 "args": [None, {"frame": {"duration": 60}}]},
                {"label": "⏸ 暂停", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}}]}
            ]
        }]
    )

    return fig


# ==================== 主程序 ====================
def main():
    st.title("🚁 地震应急无人机三维协同系统（增强版）")

    n = st.slider("无人机数量", 1, 3, 2)
    users_n = st.slider("用户数量", 20, 80, 40)
    iters = st.slider("迭代次数", 30, 120, 60)

    if st.button("开始仿真"):
        users = Users(users_n)
        obj = build_objective(n, users)

        x0 = init_pos(n, 150)

        opt = SimpleLDHAFOptimizer()
        x_opt, hist = opt.optimize(obj, x0, iters)

        # 轨迹整理
        uav_hist = []
        for i in range(n):
            traj = []
            for h in hist:
                traj.append([h[3*i], h[3*i+1], h[3*i+2]])
            uav_hist.append(traj)

        fig = create_3d(uav_hist, users)
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ 已修复：三维飞行 / 地形不消失 / 轨迹平滑")


if __name__ == "__main__":
    main()