# objects.py

import numpy as np

class AP:
    def __init__(self, ap_id, position, antennas):
        self.id = ap_id
        self.position = position  # 格式 [x, y, z]
        self.antennas = antennas
        self.assigned_ues = []   # 存储该AP服务的UE列表

    def get_channel_gain(self, ue):
        """
        返回AP与UE之间的平均信道增益（例如：基于距离的路径损耗）
        这里简单用距离倒数作为示例，你可以根据实际需要修改公式。
        """
        dist = np.linalg.norm(np.array(self.position) - np.array(ue.position))
        if dist < 1.0:
            dist = 1.0
        # 简单示例：增益与距离的倒数相关
        gain = 1.0 / dist
        return gain

    def assign_ue(self, ue):
        """
        将UE分配给此AP。
        """
        if ue not in self.assigned_ues:
            self.assigned_ues.append(ue)

    def serves(self, ue):
        """
        判断该AP是否服务UE
        """
        return (ue in self.assigned_ues)


class UE:
    def __init__(self, ue_id, position):
        self.id = ue_id
        self.position = position  # 格式 [x, y, z]
        # 可添加其他属性，如上行合并向量、导频信息等
        self.uplink_combining = None  # 后续仿真中可设置
        self.assigned_ap_ids = []     # 记录服务该UE的AP的ID

    def data_symbol(self):
        """
        生成下行数据符号。为简单起见，返回固定符号1+0j
        """
        return 1 + 0j

    # 如果需要，你也可以在此定义其他方法（例如 beta() 但这里统一由AP来实现信道增益计算）
