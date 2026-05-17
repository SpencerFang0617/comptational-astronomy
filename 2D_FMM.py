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
_K = np.arange(MAX_P)
_K1 = np.arange(1, MAX_P)

class ParticleArray:
    """
    粒子資料容器 (Structure of Arrays, SoA)。
    將所有粒子的屬性以 NumPy 陣列儲存，有利於向量化運算與內存訪問效率。
    """
    def __init__(self, xs, ys, masses):
        n = len(xs)
        # 使用複數 z = x + iy 表示 2D 空間位置
        self.pos = np.array(xs, dtype=float) + 1j * np.array(ys, dtype=float)
        self.mass = np.array(masses, dtype=float)
        self.potential = np.zeros(n, dtype=float)   # 儲存計算出的重力勢
        self.force = np.zeros(n, dtype=complex)     # 儲存計算出的重力向量 (以複數表示)
        self.n = n

class QuadtreeNode:
    """
    四元樹節點。
    用於空間分割，是 FMM 階層式計算的基礎。
    """
    __slots__ = [
        'center', 'size', 'level', 'parent', 'particle_idx', 
        'children', 'is_leaf', 'multipole_coeffs', 'local_coeffs', 
        'neighbors', 'interaction_list'
    ]
    def __init__(self, center, size, level, parent=None):
        self.center = center        # 節點中心位置 (複數)
        self.size = size            # 節點邊長
        self.level = level          # 樹的層級 (根節點為 0)
        self.parent = parent        # 父節點引用
        self.particle_idx = []      # 該節點持有的粒子索引列表
        self.children = [None] * 4  # 四個子節點
        self.is_leaf = True         # 是否為葉節點
        
        # 多極展開係數 (Multipole Expansion Coefficients) - 描述內部粒子對外部的影響
        self.multipole_coeffs = np.zeros(MAX_P, dtype=complex)
        # 局部展開係數 (Local Expansion Coefficients) - 描述外部遠方粒子對內部的影響
        self.local_coeffs = np.zeros(MAX_P, dtype=complex)
        
        self.neighbors = []         # 鄰居節點列表 (用於 P2P 計算)
        self.interaction_list = []  # 交互列表 (用於 M2L 轉換)

    def subdivide(self, pa):
        """將當前節點細分為四個子節點，並重新分配粒子。"""
        hs, qs = self.size / 2.0, self.size / 4.0
        self.is_leaf = False
        # 計算四個象限的中心
        offsets = [(-qs, qs), (qs, qs), (-qs, -qs), (qs, -qs)]
        for i, (dx, dy) in enumerate(offsets):
            self.children[i] = QuadtreeNode(self.center + complex(dx, dy), hs, self.level + 1, self)
        
        # 根據位置將粒子分配到子節點
        for idx in self.particle_idx:
            p_pos = pa.pos[idx]
            ci = (1 if p_pos.real >= self.center.real else 0) + (2 if p_pos.imag < self.center.imag else 0)
            self.children[ci].particle_idx.append(idx)
        self.particle_idx = []

# --- FMM 核心算子 (Operators) ---

def P2M(node, pa):
    """
    Particle-to-Multipole (P2M):
    將葉節點內的粒子貢獻轉化為該節點的多極展開係數。
    公式: a_0 = sum(m_i), a_k = sum(m_i * (z_i - z_c)^k)
    """
    if len(node.particle_idx) == 0: return
    idx = np.asarray(node.particle_idx)
    dz = pa.pos[idx] - node.center
    m = pa.mass[idx]
    node.multipole_coeffs[0] = m.sum()
    for k in range(1, MAX_P):
        node.multipole_coeffs[k] = np.sum(m * (dz**k))

def M2M(parent, child):
    """
    Multipole-to-Multipole (M2M):
    將子節點的多極展開係數平移並累加到父節點。
    這是向上遍歷 (Upward Pass) 的核心。
    """
    d = child.center - parent.center
    for k in range(MAX_P):
        js = np.arange(k + 1)
        # 利用二項式展開進行平移
        parent.multipole_coeffs[k] += np.sum(child.multipole_coeffs[js] * (d**(k-js)) * BINOM[k, js])

def M2L(target, source):
    """
    Multipole-to-Local (M2L):
    將遠方源節點的多極展開轉換為目標節點的局部展開。
    這是 FMM 最關鍵的一步，將「遠方群體」的影響轉化為「本地場」的近似。
    """
    d = target.center - source.center
    inv_d = 1.0 / d
    a = source.multipole_coeffs
    
    # 處理 0 階項 (對數項)
    target.local_coeffs[0] += a[0] * cmath.log(d)
    js = np.arange(1, MAX_P)
    target.local_coeffs[0] += np.sum(a[js] * ((-1.0)**js / js) * (inv_d**js))
    
    # 處理高階項
    for k in range(1, MAX_P):
        term1 = a[0] * ((-1.0)**k / k) * (inv_d**k)
        js = np.arange(1, MAX_P)
        comb = np.array([BINOM[k+j-1, j-1] for j in js])
        term2 = np.sum(a[js] * ((-1.0)**js * comb) * (inv_d**(k+js)))
        target.local_coeffs[k] += term1 + term2

def L2L(parent, child):
    """
    Local-to-Local (L2L):
    將父節點的局部展開係數平移並傳遞給子節點。
    這是向下遍歷 (Downward Pass) 的核心。
    """
    d = child.center - parent.center
    for k in range(MAX_P):
        js = np.arange(k, MAX_P)
        child.local_coeffs[k] += np.sum(parent.local_coeffs[js] * (d**(js-k)) * BINOM[js, k])

def L2P_batch(node, pa):
    """
    Local-to-Particle (L2P):
    利用節點的局部展開係數計算其內部粒子的勢能與受力。
    """
    if len(node.particle_idx) == 0: return
    idx = np.asarray(node.particle_idx)
    dz = pa.pos[idx] - node.center
    
    # 計算勢能: Phi(z) = Re( sum( b_k * (z - z_c)^k ) )
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
    p1, p2 = pa.pos[idx1], pa.pos[idx2]
    m1, m2 = pa.mass[idx1], pa.mass[idx2]
    
    if symmetric and id(idx1) == id(idx2):
        # 同一節點內的粒子兩兩交互 (利用對稱性減少一半計算量)
        for i in range(len(idx1)):
            dz = p1[i] - p1[i+1:]
            r2 = np.abs(dz)**2
            r2 = np.where(r2 < 1e-15, 1e-15, r2) # 避免除以零
            log_r = 0.5 * np.log(r2)
            inv_r2 = 1.0 / r2
            # 更新粒子 i
            pa.potential[idx1[i]] -= GRAVITATIONAL_CONSTANT * np.sum(m1[i+1:] * log_r)
            pa.force[idx1[i]] -= GRAVITATIONAL_CONSTANT * np.sum(m1[i+1:] * dz * inv_r2)
            # 更新其餘粒子 (反作用力)
            pa.potential[idx1[i+1:]] -= GRAVITATIONAL_CONSTANT * m1[i] * log_r
            pa.force[idx1[i+1:]] += GRAVITATIONAL_CONSTANT * m1[i] * dz * inv_r2
    else:
        # 不同節點間的粒子交互
        for i in range(len(idx1)):
            dz = p1[i] - p2
            r2 = np.abs(dz)**2
            r2 = np.where(r2 < 1e-15, 1e-15, r2)
            log_r = 0.5 * np.log(r2)
            inv_r2 = 1.0 / r2
            pa.potential[idx1[i]] -= GRAVITATIONAL_CONSTANT * np.sum(m2 * log_r)
            pa.force[idx1[i]] -= GRAVITATIONAL_CONSTANT * np.sum(m2 * dz * inv_r2)
            if symmetric:
                pa.potential[idx2] -= GRAVITATIONAL_CONSTANT * m1[i] * log_r
                pa.force[idx2] += GRAVITATIONAL_CONSTANT * m1[i] * dz * inv_r2

# --- FMM 求解器主類別 ---

class FMM_Solver_v3:
    def __init__(self, pa, size, max_per_node=10, max_level=5):
        self.pa = pa                # 粒子陣列
        self.size = size            # 模擬區域總尺寸
        self.max_per_node = max_per_node # 每個節點最多容納粒子數 (超過則細分)
        self.max_level = max_level  # 樹的最大深度
        self.root = None

    def build_tree(self):
        """構建四元樹並建立鄰居與交互列表。"""
        pos = self.pa.pos
        cx, cy = (pos.real.max() + pos.real.min()) / 2, (pos.imag.max() + pos.imag.min()) / 2
        self.root = QuadtreeNode(complex(cx, cy), self.size * 1.1, 0)
        for i in range(self.pa.n): self._insert(self.root, i)
        self._finalize(self.root)
        self._build_lists(self.root)

    def _insert(self, node, idx):
        """遞迴插入粒子。"""
        if node.is_leaf:
            if len(node.particle_idx) < self.max_per_node or node.level >= self.max_level:
                node.particle_idx.append(idx)
            else:
                node.subdivide(self.pa)
                self._insert(node, idx)
        else:
            p = self.pa.pos[idx]
            ci = (1 if p.real >= node.center.real else 0) + (2 if p.imag < node.center.imag else 0)
            self._insert(node.children[ci], idx)

    def _finalize(self, node):
        """將粒子索引列表轉換為 NumPy 陣列以加速後續處理。"""
        node.particle_idx = np.array(node.particle_idx, dtype=int)
        if not node.is_leaf:
            for c in node.children: self._finalize(c)

    def _is_neighbor(self, n1, n2):
        """判斷兩個節點是否在物理上相鄰。"""
        return abs(n1.center.real - n2.center.real) <= (n1.size + n2.size) * 0.51 and \
               abs(n1.center.imag - n2.center.imag) <= (n1.size + n2.size) * 0.51

    def _build_lists(self, node):
        """
        建立鄰居列表與交互列表。
        鄰居: 與自身相鄰的同層節點。
        交互列表: 父節點鄰居的子節點中，與自身不相鄰的節點。
        """
        if node.parent is None:
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
                if not pn.is_leaf:
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
    start_time = time.time()
    N, L = 500000, 10.0
    np.random.seed(42)
    # 隨機生成粒子位置與質量
    xs, ys, ms = np.random.uniform(-L/2, L/2, N), np.random.uniform(-L/2, L/2, N), np.random.uniform(0.5, 1.5, N)
    
    # --- FMM 計算 ---
    pa_fmm = ParticleArray(xs, ys, ms)
    solver = FMM_Solver_v3(pa_fmm, L)
    solver.build_tree()
    solver.run()
    print(time.time() - start_time)
    # --- 直接計算法 (Ground Truth) ---
    start_time = time.time()
    pa_dir = ParticleArray(xs, ys, ms)
    for i in range(N):
        dz = pa_dir.pos[i] - pa_dir.pos
        r2 = np.abs(dz)**2
        r2[i] = 1.0
        log_r, inv_r2 = 0.5 * np.log(r2), 1.0 / r2
        log_r[i], inv_r2[i] = 0.0, 0.0
        pa_dir.potential[i] -= GRAVITATIONAL_CONSTANT * np.dot(pa_dir.mass, log_r)
        pa_dir.force[i] -= GRAVITATIONAL_CONSTANT * np.dot(pa_dir.mass, dz * inv_r2)
    print(time.time() - start_time)
    
    # --- 誤差統計 ---
    pe = np.abs(pa_fmm.potential - pa_dir.potential) / np.abs(pa_dir.potential)
    fe = np.abs(np.abs(pa_fmm.force) - np.abs(pa_dir.force)) / np.abs(pa_dir.force)
    print(f"v3 Mean Potential Error: {np.mean(pe):.2e}")
    print(f"v3 Mean Force Error:     {np.mean(fe):.2e}")
