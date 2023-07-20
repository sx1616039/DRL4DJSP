import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, pyplot

if __name__ == "__main__":
    x_labels = ['la01', 'la02', 'la11', 'la12', 'la21', 'la22', 'la23', 'la24', 'la25', 'la26', 'la27', 'la28', 'la29', 'la30', 'la31', 'la32',
                'la33', 'la34', 'la35', 'la36', 'la37', 'la38', 'la39', 'la40', 'ta21', 'ta22', 'ta31', 'ta32',
                'ta41', 'ta42', 'ta51', 'ta52']
    # 读取数据
    trajectories = pd.read_excel("time-trajectory-data/trajectories.xls")
    time = pd.read_excel("time-trajectory-data/time.xls")

    # 设置字体
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'SimHei'  # 显示中文标签
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    palette = pyplot.get_cmap('Set1')
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 16,
    #          }

    traj = trajectories.values[:, 1:6]
    total_time = time.values[:, 1:6]
    avg_traj = np.mean(traj, axis=1)
    std_traj = np.std(traj, axis=1, dtype=np.float64)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg_traj, std_traj)))  # 上方差
    r2 = list(map(lambda x: x[0] + x[1], zip(avg_traj, std_traj)))  # 下方差

    avg_t = np.mean(total_time, axis=1)
    std_t = np.std(total_time, axis=1, dtype=np.float64)
    u_t = list(map(lambda x: x[0] - x[1], zip(avg_t, std_t)))  # 上方差
    l_t = list(map(lambda x: x[0] + x[1], zip(avg_t, std_t)))  # 下方差

    ''' 绘制曲线'''
    figure_results = plt.figure()
    axes_lines = figure_results.add_subplot(1, 1, 1)
    color = palette(0)  # 算法1颜色
    labels = ['训练轨迹数', '训练时间']
    h_traj, = axes_lines.plot(x_labels, avg_traj, 'r-')
    axes_lines.fill_between(x_labels, r1, r2, color=color, alpha=0.2)
    axes_lines.set(xlabel=u'调度实例', ylabel=u'训练轨迹数')

    axes_twins = axes_lines.twinx()
    '''绘制次坐标reward'''
    color2 = palette(1)  # 算法2颜色
    h_time, = axes_twins.plot(x_labels, avg_t, 'b-')
    axes_twins.set(ylabel=u'训练时间 / 秒')
    axes_twins.fill_between(x_labels, u_t, l_t, color=color2, alpha=0.2)
    axes_twins.legend(handles=[h_traj, h_time], labels=labels, loc='upper left')
    # plt.xlabel()
    plt.show()


