#ifndef OPERATORS_H
#define OPERATORS_H

#include "fmm.h"

/*
 * 預計算二項式係數表 C(n, k)，大小 [2*MAX_P][2*MAX_P]。
 * M2M、M2L、L2L 運算會用到。使用 Pascal's rule 迭代填表。
 */
void precompute_binomial(double binom[2 * MAX_P][2 * MAX_P]);

/*
 * Particle-to-Multipole：計算葉節點的多極展開係數。
 *   a[0] = sum(m_i)
 *   a[k] = sum(m_i * (z_i - z_c)^k),  k = 1..p
 */
void P2M(QuadtreeNode *leaf, const Particle *particles);

/*
 * Multipole-to-Multipole：子節點多極展開平移到父節點中心。
 *   d = z_child_center - z_parent_center
 *   parent.a[k] += sum_{j=0}^{k} C(k,j) * child.a[j] * d^(k-j)
 */
void M2M(QuadtreeNode *parent, const QuadtreeNode *child,
         const double binom[2 * MAX_P][2 * MAX_P]);

/*
 * Multipole-to-Local：遠場源節點的多極展開轉換為目標節點的局部展開。
 *   d = z_source_center - z_target_center
 *   0 階含 log(d) 項，高階含 (-1)^k / d^k 與二項式卷積。
 */
void M2L(const HilbertTree *tree, int node_id,
    const double complex *multipole, double complex *local_exp,
    const double binom[2*MAX_P][2*MAX_P]);


/*
 * Local-to-Local：父節點局部展開平移到子節點中心。
 *   d = z_child_center - z_parent_center
 *   child.b[k] += sum_{j=k}^{p} C(j,k) * parent.b[j] * d^(j-k)
 */
 void L2L(const HilbertTree *tree, int node_id,
    double complex *local_exp,
    const double binom[2*MAX_P][2*MAX_P]);

/*
 * Local-to-Particle：從局部展開係數計算葉節點每個粒子的勢能和力。
 *   phi = Re( sum_k b[k] * (z - z_c)^k )
 *   F   = -conj( sum_{k>=1} k * b[k] * (z - z_c)^{k-1} )
 */
void L2P(const QuadtreeNode *leaf, Particle *particles);

/*
 * Particle-to-Particle：兩個節點間的直接近場計算。
 *   phi_i += 0.5 * m_j * ln(r^2)
 *   F_i   += m_j * (z_i - z_j) / r^2
 *   r^2 下限 1e-15 防止除零。
 *   symmetric=1 時同時更新雙方粒子（避免重複計算）。
 *   node_a == node_b 時為節點內自交互。
 */
void P2P_S(const QuadtreeNode *node_a, const QuadtreeNode *node_b,
         const double binom[2*MAX_P][2*MAX_P], int symmetric);

void P2P_A(const QuadtreeNode *node_a, const QuadtreeNode *node_b,
         const double binom[2*MAX_P][2*MAX_P], int symmetric);

#endif
