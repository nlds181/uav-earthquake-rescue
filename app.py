import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="无人机协同通信平台", layout="wide")

# ==================== 优化器 ====================
class StableOptimizer:
    def optimize(self, objective_fn, x0, max_iter=100):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        v = np.zeros_like(x)
        momentum = 0.9

        for k in range(max_iter):
            grad = self._grad(objective_fn, x)
            eta = 0.1 * (0.97 ** k)
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
            150*np.exp(-((x-100)**2+(y-80)**2)/6000) +
            130*np.exp(-((x+120)**2+(y+110)**2)/7000) +
            120*np.exp(-((x-140)**2+(y+150)**2)/6500)
        )

    @staticmethod
    def surface():
        x = np.linspace(-500, 500, 70)
        y = np.linspace(-500, 500, 70)
        X, Y = np.meshgrid(x, y)
        Z = Terrain.height(X, Y)
        return X, Y, Z


# ==================== 用户 ====================
class Users:
    def __init__(self, n):
        np.random.seed(42)
        self.pos = np.random.uniform(-400, 400, (n, 2))

    def get(self):
        return self.pos


# ==================== 目标函数 ====================
def build_objective(n_uav, users):
    def obj(x):
        cost = 0.0
        u_pos = users.get()

        for i in range(n_uav):
            ux, uy, uz = x[3*i:3*i+3]

            for p in u_pos:
                d = np.hypot(ux-p[0], uy-p[1])
                cost -= 1/(1+d/50)

            # 高度变化
            target_h = 180 + 50*np.sin(ux/100) + 40*np.cos(uy/120)
            cost += abs(uz - target_h) * 0.6

            # 避障
            th = Terrain.height(ux, uy)
            if uz < th + 20:
                cost += (th + 20 - uz) * 20

        # 🚀 修复三维距离
        for i in range(n_uav):
            for j in range(i+1, n_uav):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dz = x[3*i+2] - x[3*j+2]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if dist < 50:
                    cost += (50 - dist) * 10

        return cost
    return obj


# ==================== 初始 ====================
def init_pos(n, h):
    pos = []
    for i in range(n):
        ang = 2*np.pi*i/n
        pos += [400*np.cos(ang), 400*np.sin(ang), h]
    return pos


def create_3d_animation(uav_hist, users):
    X, Y, Z = Terrain.surface()
    u_pos = users.get()
    T = len(uav_hist[0])
    n = len(uav_hist)

    colors = ['#FF3333','#33FF33','#3399FF','#FFCC33']

    def build_frame(t):
        data = []

        # ===== 地形 =====
        data.append(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.6,
            showscale=False
        ))

        # ===== 用户 =====
        data.append(go.Scatter3d(
            x=u_pos[:,0], y=u_pos[:,1], z=[50]*len(u_pos),
            mode='markers',
            marker=dict(size=4, color='gold'),
            name="用户"
        ))

        # 当前无人机位置
        curr_positions = []
        for i in range(n):
            curr_positions.append(uav_hist[i][t])

        # ===== 用户 → 最近无人机（通信链路）=====
        for p in u_pos:
            dmin = 1e9
            target = None
            for i in range(n):
                u = curr_positions[i]
                d = np.hypot(p[0]-u[0], p[1]-u[1])
                if d < dmin:
                    dmin = d
                    target = u

            data.append(go.Scatter3d(
                x=[p[0], target[0]],
                y=[p[1], target[1]],
                z=[50, target[2]],
                mode='lines',
                line=dict(color='rgba(0,200,255,0.3)', width=2),
                showlegend=False
            ))

        # ===== UAV Mesh网络 =====
        for i in range(n):
            for j in range(i+1, n):
                p1 = curr_positions[i]
                p2 = curr_positions[j]
                d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

                if d < 300:  # 通信距离
                    data.append(go.Scatter3d(
                        x=[p1[0], p2[0]],
                        y=[p1[1], p2[1]],
                        z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color='lime', width=3),
                        showlegend=False
                    ))

        # ===== UAV轨迹 + 覆盖 =====
        for i in range(n):
            traj = uav_hist[i][:t+1]
            curr = traj[-1]

            # 轨迹
            data.append(go.Scatter3d(
                x=[p[0] for p in traj],
                y=[p[1] for p in traj],
                z=[p[2] for p in traj],
                mode='lines',
                line=dict(color=colors[i%len(colors)], width=4, dash='dash'),
                showlegend=False
            ))

            # 当前点
            data.append(go.Scatter3d(
                x=[curr[0]], y=[curr[1]], z=[curr[2]],
                mode='markers',
                marker=dict(size=7, color=colors[i%len(colors)]),
                name=f"UAV-{i+1}" if t==0 else None
            ))

            # 覆盖球（简化为圆盘）
            theta = np.linspace(0, 2*np.pi, 30)
            r = 120
            cx = curr[0] + r*np.cos(theta)
            cy = curr[1] + r*np.sin(theta)
            cz = np.ones_like(cx) * curr[2]

            data.append(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode='lines',
                line=dict(color=colors[i%len(colors)], width=2),
                opacity=0.3,
                showlegend=False
            ))

        return data

    frames = [go.Frame(data=build_frame(t), name=str(t)) for t in range(T)]

    fig = go.Figure(data=build_frame(0), frames=frames)

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ 播放", "method": "animate",
                 "args": [None, {"frame": {"duration": 60, "redraw": True}}]},
                {"label": "⏸ 暂停", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}}]}
            ]
        }],
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='高度',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        ),
        height=600,
        title="🚁 无人机协同通信网络（评委震撼版）"
    )

    return fig
# ==================== 主程序 ====================
def main():
    st.title("🚁 无人机三维协同通信系统（终极稳定版）")

    n = st.slider("无人机数量",1,4,2)
    u = st.slider("用户数量",20,80,40)
    iters = st.slider("迭代次数",30,100,60)

    if st.button("开始仿真"):
        users = Users(u)
        obj = build_objective(n, users)
        x0 = init_pos(n,150)

        opt = StableOptimizer()
        x_opt, hist = opt.optimize(obj, x0, iters)

        uav_hist = []
        for i in range(n):
            traj = []
            for h in hist:
                traj.append([h[3*i],h[3*i+1],h[3*i+2]])
            uav_hist.append(traj)

        fig = create_3d_animation(uav_hist, users)
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ 已修复：多无人机 / 不消失 / 虚线轨迹 / 三维飞行")


if __name__ == "__main__":
    main()