#ifndef QUADTREE_H
#define QUADTREE_H

#include "fmm.h"

/**
 * @file quadtree.h
 * @brief 基於 Hilbert 曲線的完全四叉樹（扁平陣列，BFS 編號）。
 *
 * @section overview 概述
 *
 * 本模組將 2D 粒子依 Hilbert 曲線排序後，建構一棵完全四叉樹。
 * 樹的所有節點以 **BFS（廣度優先）順序** 存放在一維陣列中，
 * 不使用指標串接，而是透過算術公式計算父子關係。
 *
 * 提供的功能：
 *   - hilbert_tree_build()  建樹
 *   - hilbert_tree_free()   釋放記憶體
 *
 * @section node_indexing 節點編號規則
 *
 * 每一層（level）有 4^lv 個節點，BFS 連續編號：
 *
 * @code
 *   base(lv) = (4^lv - 1) / 3        // 第 lv 層第一個節點的 node_id
 *   node_id  = base(lv) + i           // 第 lv 層第 i 個節點 (i = 0 .. 4^lv - 1)
 *   total    = (4^(max_level+1) - 1) / 3
 * @endcode
 *
 * 示例（max_level = 2）：
 * @code
 *   Level 0:  node 0                              (base=0, count=1)
 *   Level 1:  node 1, 2, 3, 4                     (base=1, count=4)
 *   Level 2:  node 5, 6, 7, 8, ..., 20            (base=5, count=16)
 * @endcode
 *
 * @section parent_child 父子節點換算
 *
 * @code
 *   // 已知 level lv、層內序號 i：
 *   children:  base(lv+1) + 4*i + c      (c = 0,1,2,3)
 *   parent:    base(lv-1) + i/4
 * @endcode
 *
 * 四個子節點 c=0..3 的空間位置由 Hilbert 曲線的遞迴分解決定，
 * 與一般 Z-order 四叉樹的 NW/NE/SW/SE 不同。
 *
 * @section array_access 陣列取值方式
 *
 * 以 node_id 索引：
 *
 * | 陣列                | 型別        | 說明                                       |
 * |---------------------|------------|--------------------------------------------|
 * | node_start[id]      | int        | 該節點粒子在 sort_indices 中的起始位置       |
 * | node_end[id]        | int        | 該節點粒子在 sort_indices 中的結束位置（含）  |
 * | node_cx[id]         | double     | 節點中心 x 座標                             |
 * | node_cy[id]         | double     | 節點中心 y 座標                             |
 * | node_size[id]       | double     | 節點邊長                                   |
 *
 *以 sort_indices 索引：
 *
 * | 陣列                | 型別        | 說明                                       |
 * |---------------------|------------|--------------------------------------------|
 * | sort_x[id]         | double     | 粒子 x 座標                             |
 * | sort_y[id]         | double     | 粒子 y 座標                             |
 * | sort_mass[id]       | double     | 粒子質量                                   |
 * | sort_potential[id]   | double     | 粒子勢能                                   |
 * | sort_fx[id]         | double     | 粒子 x 方向力                             |
 * | sort_fy[id]         | double     | 粒子 y 方向力                             |
 *
 * 判斷節點是否為空：
 * @code
 *   node_start[id] > node_end[id]   // true → 空節點（無粒子）
 * @endcode
 *
 * 遍歷節點內的粒子：
 * @code
 *   for (int j = tree->node_start[id]; j <= tree->node_end[id]; j++) {
 *       double x = tree->sort_x[j];
 *       double y = tree->sort_y[j];
 *       double mass = tree->sort_mass[j];
 *       double potential = tree->sort_potential[j];
 *       double fx = tree->sort_fx[j];
 *       double fy = tree->sort_fy[j];
 *   }
 * @endcode
 *
 * 遍歷所有葉節點：
 * @code
 *   for (int i = 0; i < tree->num_leaves; i++) {
 *       int id = tree->leaf_indices[i];   // 葉節點的 node_id
 *       // ...
 *   }
 * @endcode
 */

/**
 * @brief 以 Hilbert 曲線排序粒子並建構完全四叉樹。
 *
 * 流程：
 *   1. 計算 bounding box，將 (x,y) 正規化到整數格點 [0, 2^level)
 *   2. 為每個粒子計算 Hilbert index 並排序 → sort_indices
 *   3. 逐層以二分搜尋找分割點，填入 node_start / node_end
 *   4. 利用 d2xy 反推各節點幾何中心與邊長
 *
 * @param particles    粒子陣列（唯讀）
 * @param n            粒子數量
 * @param max_per_leaf 葉節點粒子上限；當節點粒子數 <= 此值時停止分割
 * @param max_level    樹最大深度；若 <= 0 則依 max_per_leaf 自動決定
 * @return 堆積配置的 HilbertTree，由呼叫者以 hilbert_tree_free() 釋放
 */
HilbertTree *hilbert_tree_build(const Particle *particles, int n,
                                int max_per_leaf, int max_level);

/**
 * @brief 釋放 HilbertTree 的所有動態記憶體（含結構體本身）。
 *
 * @param tree 要釋放的樹；傳入 NULL 不會有任何操作
 */
void hilbert_tree_free(HilbertTree *tree);

/**
 * @brief 將Hilbert index轉換為(x,y)座標
 * @param n 格點邊長，必須是 2 的冪
 * @param d Hilbert index
 * @param x 轉換後的x座標
 * @param y 轉換後的y座標
 * @return 無
 */
void hilbert_d2xy(int n, uint64_t d, int *x, int *y);
/**
 * @brief 將(x,y)座標轉換為Hilbert index
 * @param n 格點邊長，必須是 2 的冪
 * @param x x座標
 * @param y y座標
 * @return Hilbert index
 */
uint64_t hilbert_xy2d(int n, int x, int y);
#endif
