import numpy as np
import cmath
import math
import time

# --- 核心參數設定 ---
FMM_ORDER = 12              # 多極展開的階數 (p)，階數越高精度越高，但計算量也越大
GRAVITATIONAL_CONSTANT = 1.0 # 重力常數 G
MAX_P = FMM_ORDER + 1       # 包含 0 階項的總項數

# --- 預計算組合數 (Binomial Coefficients) ---
# FMM 中的平移算子 (M2M, M2L, L2L) 頻繁使用組合數，預計算可大幅提升效能
_sz = MAX_P * 2
BINOM = np.zeros((_sz, _sz))
for _n in range(_sz):
    for _k in range(_n + 1):
        BINOM[_n, _k] = math.comb(_n, _k)

# 預計算索引陣列，用於 NumPy 向量化運算
_K  = np.arange(MAX_P)
_K1 = np.arange(1, MAX_P)

# 一些const array initialization

# Nlm (P2M要用的const array)
Nlm = np.zeros((2 * MAX_P + 1, 2 * MAX_P + 1)) 
# Anm (M2M要用的const array)
Anm = np.zeros((2 * MAX_P + 1, 2 * MAX_P + 1)) 

class ParticleArray:
    """
    粒子資料容器 (Structure of Arrays, SoA)。
    將所有粒子的屬性以 NumPy 陣列儲存，有利於向量化運算與內存訪問效率。
    """
    def __init__(self, xs, ys, zs, masses):
        n = len(xs)
        # 我用2D array 存pos[3][N] 
        self.pos = np.array([xs, ys, zs], dtype=float) # x_i = pos[0][i] ...
        self.mass = np.array(masses, dtype=float)
        self.potential = np.zeros(n, dtype=float)      # 儲存計算出的重力勢
        self.force = np.zeros((3, n), dtype=float)     # 3D用array存 
        self.n = n

class OctreeNode:

    __slots__ = [
        'center', 'size', 'level', 'parent', 'particle_idx', 
        'children', 'is_leaf', 'multipole_coeffs', 'local_coeffs', 
        'neighbors', 'interaction_list'
    ]
    def __init__(self, center, size, level, parent=None):
        self.center = np.array(center,dtype=float)          # 節點中心位置 (x, y, z)
        self.size = size                                    # 節點邊長
        self.level = level                                  # 樹的層級 (根節點為 0)
        self.parent = parent                                # 父節點引用
        self.particle_idx = []                              # 該節點持有的粒子索引列表
        self.children = [None] * 8                          # 八個子節點
        self.is_leaf = True                                 # 是否為葉節點
        
        # 多極展開係數 (Multipole Expansion Coefficients) - 描述內部粒子對外部的影響
        self.multipole_coeffs = np.zeros((MAX_P + 1, 2 * MAX_P + 1), dtype=complex)
        # 局部展開係數 (Local Expansion Coefficients) - 描述外部遠方粒子對內部的影響
        self.local_coeffs = np.zeros((MAX_P + 1, 2 * MAX_P + 1), dtype=complex)
        
        self.neighbors = []         # 鄰居節點列表 (用於 P2P 計算)
        self.interaction_list = []  # 交互列表 (用於 M2L 轉換)

    def subdivide(self, pa):
        """將當前節點細分為八個子節點，並重新分配粒子。"""
        hs, qs = self.size / 2.0, self.size / 4.0

        self.is_leaf = False
        # 計算八個象限的中心
        offsets = [(-qs, -qs, -qs), (-qs, -qs, qs), (-qs, qs, -qs), (-qs, qs, qs), 
                   (qs, -qs, -qs ), (qs, -qs, qs ), (qs, qs, -qs ), (qs, qs, qs ) ]

        for i, (dx, dy, dz) in enumerate(offsets):
            self.children[i] = OctreeNode(self.center + np.array([dx, dy, dz]), hs, self.level + 1, self)
        
        # 根據位置將粒子分配到子節點
        for idx in self.particle_idx:
            p_pos = pa.pos[:, idx] # [x_idx, y_idx, z_idx]
            # 0 => ---, 4 => +--, 2 => -+-, 1 => --+, 
            # 3 => -++, 5 => +-+, 6 => ++-, 7 => +++
            # 用2進位 + => 1, - => 0
            ci = 4 * (p_pos[0] >= self.center[0]) + \
                 2 * (p_pos[1] >= self.center[1]) + \
                 1 * (p_pos[2] >= self.center[2])
        
            self.children[ci].particle_idx.append(idx)
        self.particle_idx = []

# 一些operators 會用到的function 

def build_const_Array():
    # 建一些P2M, M2M要用的表 
    for l in range(2 * MAX_P + 1):
        for m in range(l + 1):
            Nlm[l, m] = np.sqrt(math.factorial(l - m) / math.factorial(l + m))
            Anm[l, m] = (-1)**l / np.sqrt(math.factorial(l - m) * math.factorial(l + m))

def get_LegendreP(sintheta, costheta, Pscale):
    sintheta = np.atleast_1d(sintheta)
    costheta = np.atleast_1d(costheta)
    size = len(sintheta)
    # 算P_mn(phi')
    # (1) P_(m-1)(m-1)       => P_mm     = -(2m - 1) * sin(theta') * P_(m-1)(m-1)(cos(theta'))
    # (2) P_mm               => P_m(m+1) =  (2m + 1) * cos(theta') * P_mm(cos(theta'))
    # (3) P_m(l-1), P_m(l-2) => P_ml     =  (2l - 1)/(l - m) * cos(theta') * P_m(l-1)(cos(theta'))
    #                                    -  (l + m - 1)/(l - m) * P_m_(l-2)(cos(theta'))
    # 從P_00 = 1 開始

    Plm = np.zeros((Pscale, Pscale, size))  
    Plm[0, 0, :] = 1.0

    for m in range(0, Pscale - 1):
        # (1)
        Plm[m + 1, m + 1, :] = -(2*m + 1) * sintheta * Plm[m, m, :]
        # (2)
        Plm[m + 1, m, :]     =  (2*m + 1) * costheta * Plm[m, m, :]
    
    for l in range(2, Pscale):
        # (3)
        for m in range(l - 1):
            Plm[l, m, :] = (2*l - 1) / (l - m) * costheta * Plm[l-1, m, :] \
                        - (l + m - 1) / (l - m) * Plm[l-2, m, :] 
    if(size == 1):
        return Plm[:, :, 0]

    return Plm

def get_SphHarmoY(Plm, exp_i_phi, Yscale):
    exp_i_phi = np.atleast_1d(exp_i_phi)
    size = len(exp_i_phi)

    Plm = np.asarray(Plm).reshape(Yscale, Yscale, -1)
    Ylm = np.zeros((Yscale, Yscale, size), dtype = complex)
    
    for l in range(Yscale):
        exp_now = np.ones(size, dtype = complex)
        for m in range(l + 1):
            Ylm[l, m, :] = Nlm[l, m] * Plm[l, m, :] * exp_now
            exp_now *= exp_i_phi

    if(size == 1):
        return Ylm[:, :, 0]
    return Ylm

def get_inf(pos1, pos2):
    pos1, pos2 = np.asarray(pos1).reshape(3, -1), np.asarray(pos2).reshape(3, -1) # 讓array 大小固定

    ds  = pos1 - pos2                               # displacement (這是一個array, 大小取決於idx = ?)
    dxy = np.sqrt(np.sum(ds[:2]**2, axis = 0))      # distance on xy plane s' 
    dxy = np.where(dxy < 1e-15, 1e-15, dxy)         # prevent it explodes
    rs  = np.sqrt(np.sum(ds**2, axis = 0))          # r'
    rs  = np.where(rs < 1e-15, 1e-15, rs)           # prevent it exlpodes

    costheta = ds[2] / rs 
    sintheta = np.sqrt(1 - costheta**2)
    exp_iphi = (ds[0] + 1j * ds[1]) / dxy           # exp(iphi) = cos(phi) + i * sin(phi)

    return np.squeeze(ds), np.squeeze(rs), np.squeeze(dxy), np.squeeze(costheta), np.squeeze(sintheta), np.squeeze(exp_iphi)

# --- FMM 核心算子 (Operators) ---

def P2M(node, pa):
    # Particle to Multipole 將葉節點內的粒子貢獻轉化為該節點的多極展開係數
    # 流程大概是 算particle -> center 的(r',theta',phi')
    # (事實上我們只需要sin,cos(theta'), exp(i * m * phi'))
    # 現在的展開係數是M_ml 要算 M_lm 要有 Ylm(theta',phi') 要有Ylm就要有 Plm(cos(theta')), Nlm 
    # 此外Nlm 是Ylm 前面的const 因為他是一堆階乘算起來很花時間 所以這裡我先在外面算好再丟進來 
    # m <= l <= MAX_P 
   
    # 算 (r',theta',phi')
    N = len(node.particle_idx)
    if N == 0: return

    idx = np.asarray(node.particle_idx)
    mas = pa.mass[idx]
    ds, rs, dxy, costheta, sintheta, exp_iphi = get_inf(pa.pos[:, idx], node.center.reshape(3, 1)) # pos1, pos2 (ds = pos2 -> pos1(pos1 - pos2))
    
    # 算P_mn(phi')
    Plm = get_LegendreP(sintheta, costheta, MAX_P + 1) # sintheta, costheta, Pscale 
    
    # 算Ylm(theta',phi')
    Ylm = get_SphHarmoY(Plm, exp_iphi, MAX_P + 1) # size, expiphi 
    
    # 算Mlm
    Mlm = np.zeros((MAX_P + 1, 2 * MAX_P + 1), dtype = complex) # m 從 -l 到 l  
    for l in range(MAX_P + 1):
        num_indep_of_m = mas * rs**l
        for m in range(l + 1):
            # -l, -l+1, ..., -1, 0, 1, 2 .... l
            Mlm[l, MAX_P + m] = np.sum(num_indep_of_m * np.conj(Ylm[l, m, :])) # Y_l(-m) = Y_lm*
            if m: Mlm[l, MAX_P - m] = np.sum(num_indep_of_m * Ylm[l, m, :])

    node.multipole_coeffs = Mlm

def M2M(parent, child):
    # Multipole-to-Multipole 展開係數平移並累加到父節點
    # 有 O(p^4) 很無腦的寫法 <-- 我這裡用的
    # 這個方法一樣要建表格 Anm , 一樣寫在外面
    # 還有O(p^3) exp 的方法  可能之後會改用這個

    # 算 (r',theta',phi')
    ds, rs, dxy, costheta, sintheta, exp_iphi = get_inf(child.center, parent.center) # pos1, pos2 (ds = pos2 -> pos1(pos1 - pos2))

    # 算Plm
    Plm = get_LegendreP(sintheta, costheta, MAX_P + 1)

    # 算Ylm(theta',phi')
    Ylm = get_SphHarmoY(Plm, exp_iphi, MAX_P + 1) # Plm, exp_iphi, Ysclae

    # 算M'lm 
    Olm = child.multipole_coeffs
    Mjk = np.zeros((MAX_P + 1, 2 * MAX_P + 1), dtype = complex)
    for j in range(MAX_P + 1):
        for k in range(-j, j + 1):
            add_num = 0.0 + 0.0j 
            for n in range(j + 1):
                for m in range(-n, n + 1):
                    # j - n  > 0 
                    # 但是 k - m 有些 < 0 
                    # 我們上面的P2M 有定義 Olm(# -l, -l+1, ..., -1, 0, 1, 2 .... l)
                
                    # Olm[j - n, k - m] = 0 不再這裡切掉的話會有redundent的計算
                    if(abs(k - m) > (j - n)):
                        continue
                    # Y +m -m 定義不同
                    if(m > 0): flag = np.conj(Ylm[n, m]) # m > 0 => -m < 0 Yn(-m) = *Ynm
                    else: flag = Ylm[n, -m]

                    add_num += (Olm[j - n, MAX_P + k - m] * (1j)**(abs(k)-abs(m)-abs(k-m))  \
                                       *  Anm[n, abs(m)] * Anm[j - n, abs(k - m)] * rs**n * (flag))
                
            Mjk[j, MAX_P + k] += add_num / Anm[j, abs(k)]
    parent.multipole_coeffs = Mjk

def M2L(target, source):
    # 丟進來兩個node
    """
    Multipole-to-Local (M2L):
    將遠方源節點的多極展開轉換為目標節點的局部展開。
    這是 FMM 最關鍵的一步，將「遠方群體」的影響轉化為「本地場」的近似。
    """
    # pos1, pos2 (ds = pos2 -> pos1(pos1 - pos2))
    ds, rs, dxy, costheta, sintheta, exp_iphi = get_inf(source.center, target.center)
    
    Plm = get_LegendreP(sintheta, costheta, 2 * MAX_P + 1)
    Ylm = get_SphHarmoY(Plm, exp_iphi, 2 * MAX_P + 1)

    Onm = source.multipole_coeffs
    Ljk = np.zeros((MAX_P + 1, 2 * MAX_P + 1), dtype = complex)

    for j in range(MAX_P + 1):
        for k in range(-j, j + 1):
            add_num = 0.0 + 0.0j

            for n in range(MAX_P + 1):
                for m in range(-n, n + 1):
                    # Y +m -m 定義不同
                    if(m - k < 0): flag = np.conj(Ylm[j + n, -m + k]) # m > 0 => -m < 0 Yn(-m) = *Ynm
                    else: flag = Ylm[j + n, m - k]

                    add_num += Onm[n, MAX_P + m] * (1j)**(abs(k - m) - abs(k) - abs(m)) * Anm[n, abs(m)] \
                             * Anm[j , abs(k)] * flag / ((-1)**n * rs**(j + n + 1) * Anm[j + n, abs(m - k)]) # Anm 會跑到2P

            Ljk[j, MAX_P + k] += add_num
    target.local_coeffs = Ljk

def L2L(parent, child):
    """
    Local-to-Local (L2L):
    將父節點的局部展開係數平移並傳遞給子節點。
    這是向下遍歷 (Downward Pass) 的核心。
    """
    # pos1, pos2 (ds = pos2 -> pos1(pos1 - pos2))
    ds, rs, dxy, costheta, sintheta, exp_iphi = get_inf(child.center, parent.center)
    
    Plm = get_LegendreP(sintheta, costheta, MAX_P + 1)
    Ylm = get_SphHarmoY(Plm, exp_iphi, MAX_P + 1)

    Onm = parent.local_coeffs
    Ljk = np.zeros((MAX_P + 1, 2 * MAX_P + 1), dtype = complex)

    for j in range(MAX_P + 1):
        for k in range(-j, j + 1):
            add_num = 0.0 + 0.0j
            for n in range(j, MAX_P + 1):
                rs_now = rs**(n - j)
                for m in range(-n, n + 1):
                    if(m - k > 0) : flag = Ylm[n - j, m - k]
                    else: flag = np.conj(Ylm[n - j, k - m])

                    add_num += Onm[n, MAX_P + m] * (1.0j)**(abs(m) - abs(m - k) - abs(k)) \
                                * Anm[n - j, abs(m - k)] * Anm[j, abs(k)] * flag * rs_now \
                                / ((-1)**(n + j) * Anm[n, abs(m)]) 
            Ljk[j, MAX_P + k] += add_num
    child.local_coeffs += Ljk

def L2P_batch(node, pa):
    """
    Local-to-Particle (L2P):
    利用節點的局部展開係數計算其內部粒子的勢能與受力。
    """
    if len(node.particle_idx) == 0: return
    idx = np.asarray(node.particle_idx)
    dz = pa.pos[:, idx] - node.center.reshape(3, 1)
    
    # 計算勢能
    pot = np.zeros(len(idx), dtype=complex)
    for k in range(MAX_P):
        pot += node.local_coeffs[k] * (dz**k)
    pa.potential[idx] -= GRAVITATIONAL_CONSTANT * pot.real
    
    # 計算受力 (勢能的負梯度): F = -conj( dPhi/dz )
    force = np.zeros(len(idx), dtype=complex)
    for k in range(1, MAX_P):
        force += k * node.local_coeffs[k] * (dz**(k-1))
    pa.force[idx] -= GRAVITATIONAL_CONSTANT * force.conjugate()

def P2P_batch(idx1, idx2, pa, symmetric=True):
    """
    Particle-to-Particle (P2P):
    直接計算近場粒子間的交互作用。
    對於不滿足遠場近似條件的粒子，必須使用直接計算法以保證精度。
    """

    if len(idx1) == 0 or len(idx2) == 0: return
    p1, p2 = pa.pos[:,  idx1], pa.pos[;,  idx2] 
    m1, m2 = pa.mass[:, idx1], pa.mass[;, idx2]
    
    if symmetric and id(idx1) == id(idx2):
        # 同一節點內的粒子兩兩交互 (利用對稱性減少一半計算量)
        for i in range(len(idx1)):
            dz = p1[:, i].reshape(3, 1) - p1[:, i+1:]
            r2 = np.sum(dz * dz, axis = 0)
            r2 = np.where(r2 < 1e-15, 1e-15, r2) # 避免除以零
            inv_r  = 1.0 / np.sqrt(r2)
            inv_r3 = 1.0 / (inv_r)**3

            # 更新粒子 i
            pa.potential[idx1[i]] -= GRAVITATIONAL_CONSTANT * np.sum(m1[i+1:] * inv_r, axis = 0)
            pa.force[idx1[i]]     -= GRAVITATIONAL_CONSTANT * np.sum(m1[i+1:] * dz * inv_r3, axis = 1)
            # 更新其餘粒子 (反作用力)
            pa.potential[idx1[i+1:]] -= GRAVITATIONAL_CONSTANT * m1[i] * inv_r
            pa.force[idx1[i+1:]]     += GRAVITATIONAL_CONSTANT * m1[i] * dz * inv_r3
    else:
        # 不同節點間的粒子交互
        for i in range(len(idx1)):
            dz = p1[:, i].reshape(3, 1) - p2  
            r2 = np.sum(dz**2, axis = 0)        
            r2 = np.where(r2 < 1e-15, 1e-15, r2)
            inv_r  = 1.0 / np.sqrt(r2)
            inv_r3 = (inv_r)**3

            pa.potential[idx1[i]] -= GRAVITATIONAL_CONSTANT * np.sum(m2 * inv_r)
            pa.force[:, idx1[i]]  -= GRAVITATIONAL_CONSTANT * np.sum(m2 * dz * inv_r3, axis=1)

            if symmetric:
                pa.potential[idx2] -= GRAVITATIONAL_CONSTANT * m1[i] * inv_r
                pa.force[:, idx2]  += GRAVITATIONAL_CONSTANT * m1[i] * dz * inv_r3

# --- FMM 求解器主類別 ---

class FMM_Solver_v3:
    def __init__(self, pa, size, max_per_node=10, max_level=5):
        self.pa = pa                        # 粒子陣列
        self.size = size                    # 模擬區域總尺寸
        self.max_per_node = max_per_node    # 每個節點最多容納粒子數 (超過則細分)(useless in non adap) 
        self.max_level = max_level          # 樹的最大深度
        self.root = None

    def build_tree(self):
        """構建四元樹並建立鄰居與交互列表。"""
        pos = self.pa.pos                   # pos = pos[3][N]

        # get center pos 
        cx = (pos[0].max() + pos[0].min())/2 
        cy = (pos[1].max() + pos[1].min())/2
        cz = (pos[2].max() + pos[2].min())/2

        self.root = OctreeNode(np.array([cx, cy, cz]), self.size * 1.1, 0)
        for i in range(self.pa.n): self._insert(self.root, i)
        self._finalize(self.root)
        self._build_lists(self.root)

    def _insert(self, node, idx):
        """遞迴插入粒子。"""
        if node.is_leaf:
            if(node.level >= self.max_level):
                node.particle_idx.append(idx)
            else:
                node.subdivide(self.pa)
                self._insert(node, idx)
        else:
            # 0 => ---, 4 => +--, 2 => -+-, 1 => --+, 
            # 3 => -++, 5 => +-+, 6 => ++-, 7 => +++
            # 用2進位 + => 1, - => 0
            ci = 4 * (self.pa.pos[0][idx] >= node.center[0]) + \
                 2 * (self.pa.pos[1][idx] >= node.center[1]) + \
                 1 * (self.pa.pos[2][idx] >= node.center[2])
        
            self._insert(node.children[ci], idx)

    def _finalize(self, node):
        """將粒子索引列表轉換為 NumPy 陣列以加速後續處理。"""
        node.particle_idx = np.array(node.particle_idx, dtype=int)
        if not node.is_leaf:
            for c in node.children: self._finalize(c)

    def _is_neighbor(self, n1, n2):
        return np.max(np.abs(n1.center - n2.center)) <= (n1.size + n2.size) * 0.51

    def _build_lists(self, node):
        # list 裡存node 
        """
        建立鄰居列表與交互列表。
        鄰居: 與自身相鄰的同層節點。
        交互列表: 父節點鄰居的子節點中，與自身不相鄰的節點。
        """
        if node.parent is None:
            # level 0 
            node.neighbors = [node]
        else:
            # 尋找鄰居
            for pn in node.parent.neighbors:
                if pn.is_leaf:
                    if self._is_neighbor(node, pn): node.neighbors.append(pn)
                else:
                    for c in pn.children:
                        if self._is_neighbor(node, c): node.neighbors.append(c)
            # 建立交互列表 (M2L 的對象)
            for pn in node.parent.neighbors:
                if(not pn.is_leaf):
                    for c in pn.children:
                        if not self._is_neighbor(node, c): node.interaction_list.append(c)
        if not node.is_leaf:
            for c in node.children: self._build_lists(c)

    def run(self):
        """執行 FMM 完整流程。"""
        self._upward(self.root)    # 1. 向上遍歷: P2M -> M2M
        self._downward(self.root)  # 2. 向下遍歷: M2L -> L2L
        self._evaluate(self.root)  # 3. 最終評估: L2P + P2P

    def _upward(self, node):
        if node.is_leaf: P2M(node, self.pa)
        else:
            for c in node.children: self._upward(c)
            for c in node.children: M2M(node, c)

    def _downward(self, node):
        for src in node.interaction_list: M2L(node, src)
        if not node.is_leaf:
            for c in node.children:
                L2L(node, c)
                self._downward(c)

    def _evaluate(self, node):
        if node.is_leaf:
            L2P_batch(node, self.pa)
            # 處理近場 P2P
            P2P_batch(node.particle_idx, node.particle_idx, self.pa, symmetric=True)
            for nb in node.neighbors:
                if nb is not node and nb.is_leaf:
                    if id(node) < id(nb): P2P_batch(node.particle_idx, nb.particle_idx, self.pa, symmetric=True)
        else:
            for c in node.children: self._evaluate(c)

# --- 測試與驗證 ---

if __name__ == "__main__":
    N, L = 500, 10.0
    np.random.seed(42)
    # 隨機生成粒子位置與質量
    xs, ys, ms = np.random.uniform(-L/2, L/2, N), np.random.uniform(-L/2, L/2, N), np.random.uniform(0.5, 1.5, N)
    
    # --- FMM 計算 ---
    build_const_Array()
    pa_fmm = ParticleArray(xs, ys, ms)
    solver = FMM_Solver_v3(pa_fmm, L)
    solver.build_tree()
    solver.run()
    
    # --- 直接計算法 (Ground Truth) ---
    pa_dir = ParticleArray(xs, ys, ms)
    for i in range(N):
        dz = pa_dir.pos[i] - pa_dir.pos
        r2 = np.abs(dz)**2
        r2[i] = 1.0
        log_r, inv_r2 = 0.5 * np.log(r2), 1.0 / r2
        log_r[i], inv_r2[i] = 0.0, 0.0
        pa_dir.potential[i] -= GRAVITATIONAL_CONSTANT * np.dot(pa_dir.mass, log_r)
        pa_dir.force[i] -= GRAVITATIONAL_CONSTANT * np.dot(pa_dir.mass, dz * inv_r2)
    
    # --- 誤差統計 ---
    pe = np.abs(pa_fmm.potential - pa_dir.potential) / np.abs(pa_dir.potential)
    fe = np.abs(np.abs(pa_fmm.force) - np.abs(pa_dir.force)) / np.abs(pa_dir.force)
    print(f"v3 Mean Potential Error: {np.mean(pe):.2e}")
    print(f"v3 Mean Force Error:     {np.mean(fe):.2e}")