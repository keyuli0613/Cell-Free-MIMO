# pilot_assignment.py
import numpy as np

def assign_pilots(ue_list, ap_list, beta_matrix, L=5, tau_p=None):
    """
    基于大尺度衰落选择前 L 个最强 AP，并为 UE 分配导频，
    同时形成动态合作聚类 (DCC)。
    
    输入参数：
      ue_list: UE 对象列表，每个 UE 对象需包含 id 属性
      ap_list: AP 对象列表，每个 AP 对象需包含 id 属性
      beta_matrix: 一个形状为 (NUM_AP, NUM_UE) 的大尺度衰落系数矩阵，
                   beta_matrix[l, k] 表示 AP l 与 UE k 之间的衰落系数
      L: 对每个 UE 选择前 L 个最强 AP（默认 L=5）
      tau_p: 导频数（如果未指定，则默认为 UE 数量，即每个 UE 可获得唯一导频）
    
    输出：
      pilot_assignments: 字典 {ue_id: pilot_index}，记录每个 UE 的导频分配结果
      dcc: 字典 {ue_id: [ap_id, ...]}，记录每个 UE 被哪些 AP 服务（形成动态合作聚类）
    """
    # 若未指定 tau_p，则默认导频数为 UE 数量
    if tau_p is None:
        tau_p = len(ue_list)
    
    # 1. 对于每个 UE，根据 beta_matrix 选取前 L 个最强 AP，形成动态合作聚类 (DCC)
    dcc = {}
    for ue in ue_list:
        # beta_matrix[:, ue.id] 取出所有 AP 对该 UE 的大尺度衰落系数，
        # np.argsort(...)[::-1] 得到降序排列的 AP 索引，取前 L 个
        ranked_aps = np.argsort(beta_matrix[:, ue.id])[::-1][:L]
        # 将 AP 对象中 id 在 ranked_aps 中的加入该 UE 的 DCC
        dcc[ue.id] = [ap.id for ap in ap_list if ap.id in ranked_aps]
    
    # 2. 导频分配（使用贪心算法避免同一聚类内的 UE 分配相同导频）
    pilot_assignments = {}
    # 这里可用的导频数为 tau_p（如果 tau_p 小于 UE 数量，则会进行导频复用）
    available_pilots = list(range(tau_p))
    for ue in ue_list:
        used_pilots = set()
        # 遍历其他 UE，如果其 DCC 与当前 UE 的 DCC 有交集且该 UE 已经分配了导频，
        # 则认为它们之间可能相互干扰，从而将该导频视为不可用
        for neighbor_ue in ue_list:
            if neighbor_ue.id in pilot_assignments:
                # 如果两个 UE 的 DCC 存在交集，则视为邻居
                if len(set(dcc[ue.id]) & set(dcc[neighbor_ue.id])) > 0:
                    used_pilots.add(pilot_assignments[neighbor_ue.id])
        # 从可用导频中选择一个未被邻居占用的导频进行分配
        for pilot in available_pilots:
            if pilot not in used_pilots:
                pilot_assignments[ue.id] = pilot
                break
        # 如果所有导频都被占用，则可以随机分配或选择最小干扰的导频
        if ue.id not in pilot_assignments:
            pilot_assignments[ue.id] = available_pilots[np.argmin(list(used_pilots))] if used_pilots else available_pilots[0]
    
    return pilot_assignments, dcc
