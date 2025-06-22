import numpy as np

keypoints_yolo_key2position = {
    0: "Nose",
    1: "Left-eye",
    2: "Right-eye",
    3: "Left-ear",
    4: "Right-ear",
    5: "Left-shoulder",
    6: "Right-shoulder",
    7: "Left-elbow",
    8: "Right-elbow",
    9: "Left-wrist",
    10: "Right-wrist",
    11: "Left-hip",
    12: "Right-hip",
    13: "Left-knee",
    14: "Right-knee",
    15: "Left-ankle",
    16: "Right-ankle"
}

class Graph():
    def __init__(self,
                 layout='yolo',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'yolo':
            self.num_node = 17  # Adjusted to match the keypoints dictionary
            self_link = [(i, i) for i in range(self.num_node)]
            
            # Updated skeletal connections based on the 17 keypoints
            neighbor_link = [
                (4, 3), (3, 2),  # Right ear to right eye, right eye to nose
                (7, 6), (6, 5),  # Left elbow to left shoulder, left shoulder to left hip
                (13, 12), (12, 11),  # Left knee to left hip, left hip to right hip
                (10, 9), (9, 8),  # Right wrist to right elbow, right elbow to right shoulder
                (11, 5),  # Left hip to left shoulder
                (8, 2),  # Right elbow to right eye
                (5, 1),  # Left shoulder to left eye
                (2, 1),  # Right eye to left eye
                (0, 1),  # Nose to left eye
                (15, 0), (14, 0)  # Left ankle to nose, Right ankle to nose
            ]
            # neighbor_link = [
            #     (0, 1), (0, 8),  (0, 10),# Right ear to right eye, right eye to nose
            #     (1, 0),  # Left elbow to left shoulder, left shoulder to left hip
            #     (2, 3), (2, 8),  (2, 11),# Left knee to left hip, left hip to right hip
            #     (3, 2),  # Right wrist to right elbow, right elbow to right shoulder
            #     (4, 5),  (4, 9), (4, 10),# Left hip to left shoulder
            #     (5, 4),  # Right elbow to right eye
            #     (6, 7), (6, 9),  (6, 11),   # Left shoulder to left eye
            #     (7, 6),  # Right eye to left eye
            #     (8, 0),  (8, 2), (8, 10), (8, 11), # Nose to left eye
            #     (9, 10), (9, 11), (9, 4), (9, 6), 
            #     (10, 9),(10, 8), (10, 0),  (10,4),# Left ankle to nose, Right ankle to nose
            #     (11, 2), (11, 8) ,(11,9), (11, 6)
            # ]
            self.edge = self_link + neighbor_link
            self.center = 1  # Left-eye as the center

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
