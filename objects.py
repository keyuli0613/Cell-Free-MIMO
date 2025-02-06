# objects.py

class AP:
    def __init__(self, ap_id, position, antennas):
        self.id = ap_id
        self.position = position  # 新增
        self.antennas = antennas
        self.assigned_ues = []

    def assign_ue(self, ue):
        if ue not in self.assigned_ues:
            self.assigned_ues.append(ue)

    def serves(self, ue):
        return ue in self.assigned_ues


class UE:
    def __init__(self, ue_id, position):
        self.id = ue_id
        self.position = position
        self.assigned_ap_ids = []

    def data_symbol(self):
        return 1+0j  # 简化
