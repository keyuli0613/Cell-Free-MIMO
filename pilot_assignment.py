# pilot_assignment.py

import numpy as np

def assign_pilots(ue_list, ap_list, beta_matrix, L=5):
    """基于大尺度衰落选择前L个AP，并分配导频"""
    dcc = {}
    for ue in ue_list:
        # 选择前L个最强AP
        ranked_aps = np.argsort(beta_matrix[:, ue.id])[::-1][:L]
        dcc[ue.id] = [ap.id for ap in ap_list if ap.id in ranked_aps]
    
    # 导频分配（贪心算法避免冲突）
    pilot_assignments = {}
    available_pilots = list(range(len(ue_list)))
    for ue in ue_list:
        used_pilots = set()
        for neighbor_ue in ue_list:
            if len(set(dcc[ue.id]) & set(dcc[neighbor_ue.id])) > 0 and neighbor_ue.id in pilot_assignments:
                used_pilots.add(pilot_assignments[neighbor_ue.id])
        for pilot in available_pilots:
            if pilot not in used_pilots:
                pilot_assignments[ue.id] = pilot
                break
    return pilot_assignments, dcc