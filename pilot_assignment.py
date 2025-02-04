# pilot_assignment.py
import numpy as np
from config import TAU_P

def assign_pilots(ue_list, ap_list):
    """
    基于书中算法4.1为UE分配导频，并形成合作聚类 (DCC)。
    
    参数：
      ue_list: UE对象列表，每个UE对象应包含属性如id、位置等
      ap_list: AP对象列表，每个AP对象应包含属性如id、位置、服务UE集合等，
               并实现方法 get_channel_gain(ue) 来返回AP与指定UE之间的平均信道增益，
               以及 assign_ue(ue) 方法，将UE加入AP的服务列表。
    
    返回：
      pilot_assignments: 字典 {ue_id: pilot_index}，记录每个UE被分配的导频
      dcc: 字典 {ue_id: list_of_ap_ids}，记录每个UE对应的合作AP集合（DCC）
    """
    pilot_assignments = {}
    dcc = {}
    
    # Step 1: 对前 TAU_P 个 UE 分配互相正交的导频
    for k in range(min(TAU_P, len(ue_list))):
        pilot_assignments[ue_list[k].id] = k  # 导频索引从 0 到 TAU_P-1
        dcc[ue_list[k].id] = []
    
    # Step 2: 对剩余的 UE 采用贪心法分配导频
    for k in range(TAU_P, len(ue_list)):
        ue = ue_list[k]
        # 找出UE与所有AP中信道增益最大的AP
        # 这里调用 AP 对象的 get_channel_gain(ue) 方法
        best_ap = max(ap_list, key=lambda ap: ap.get_channel_gain(ue))
        
        # 在 best_ap 上，对每个导频 t 计算已有UE引入的干扰：
        # 对于每个导频 t，遍历所有已经被分配该导频的UE，累加它们与 best_ap 之间的信道增益。
        pilot_interference = {}
        for t in range(TAU_P):
            pilot_interference[t] = sum(
                [best_ap.get_channel_gain(other_ue)
                 for other_ue in ue_list
                 if pilot_assignments.get(other_ue.id, None) == t]
            )
        # 选择干扰最小的导频
        best_pilot = min(pilot_interference, key=pilot_interference.get)
        pilot_assignments[ue.id] = best_pilot
        dcc[ue.id] = []  # 初始化该UE的AP集合（后续Step 3更新）
    
    # # Step 3: 每个AP根据自己的局部信息选择服务的UE
    # for ap in ap_list:
    #     for t in range(TAU_P):
    #         # 对于AP，选出其视野中，所有被分配了导频 t 的UE
    #         candidate_ues = [ue for ue in ue_list if pilot_assignments.get(ue.id, None) == t]
    #         if candidate_ues:
    #             # 选择在该AP上信道增益最大的UE
    #             best_ue = max(candidate_ues, key=lambda ue: ap.get_channel_gain(ue))
    #             # 将该UE分配给该AP
    #             ap.assign_ue(best_ue)
    #             # 在该UE的DCC中加入该AP的ID
    #             dcc[best_ue.id].append(ap.id)
    
    # return pilot_assignments, dcc
    
    # Step 3: 每个AP本地选择UE
    for ap in ap_list:
        for t in range(TAU_P):
            # 找到导频t对应的UE
            candidate_ues = [ue for ue in ue_list if pilot_assignments.get(ue.id, None) == t]
            if candidate_ues:
                # 在本AP上挑选信道增益最大的UE
                best_ue = max(candidate_ues, key=lambda u: ap.get_channel_gain(u))
                ap.assign_ue(best_ue)
                # 确保 best_ue.id 在 dcc[best_ue.id] 不重复添加
                if ap.id not in dcc[best_ue.id]:
                    dcc[best_ue.id].append(ap.id)
    
    return pilot_assignments, dcc