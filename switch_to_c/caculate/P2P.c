#include <math.h>
#include "fmm.h"
#include "operators.h"

/*
 * P2P：兩個不同節點之間的粒子-粒子直接近場計算。
 *   對 node_a 中的每個粒子 i 與 node_b 中的每個粒子 j：
 *       phi_i += G * m_j * 0.5 * ln(r^2)
 *       F_i   -= G * m_j * (r_i - r_j) / r^2
 *   r^2 下限 1e-15 防止除零。
 *   若 is_symmetric == true，同時把對稱的貢獻加到 node_b 的粒子上，
 *   避免呼叫者再呼叫一次 P2P(node_b, node_a)。
 */
void P2P(const HilbertTree *tree, int node_a_id, int node_b_id, bool is_symmetric)
{
    int a_s = tree->node_start[node_a_id];
    int a_e = tree->node_end[node_a_id];
    int b_s = tree->node_start[node_b_id];
    int b_e = tree->node_end[node_b_id];
    if (a_s > a_e || b_s > b_e) return;

    for (int i = a_s; i <= a_e; i++) {
        double xi = tree->sort_x[i];
        double yi = tree->sort_y[i];
        double mi = tree->sort_mass[i];

        double pi_acc  = 0.0;
        double fxi_acc = 0.0;
        double fyi_acc = 0.0;

        for (int j = b_s; j <= b_e; j++) {
            double dx = xi - tree->sort_x[j];
            double dy = yi - tree->sort_y[j];
            double r2 = dx * dx + dy * dy;
            if (r2 < 1e-15) r2 = 1e-15;

            double mj    = tree->sort_mass[j];
            double pot   = log(r2);
            double invr2 = 1.0 / r2;
            double fxij  = dx * invr2;
            double fyij  = dy * invr2;

            pi_acc  += G_CONST * mj * pot*0.5;
            /* fxij = (xi - xj) / r^2, fyij = (yi - yj) / r^2
               因為力 F = -G m (r_i - r_j) / r^2 */
            fxi_acc -= G_CONST * mj * fxij;
            fyi_acc -= G_CONST * mj * fyij;

            if (is_symmetric) {
                tree->sort_potential[j] += G_CONST * mi * pot*0.5;
                tree->sort_fx[j]        += G_CONST * mi * fxij;
                tree->sort_fy[j]        += G_CONST * mi * fyij;
            }
        }

        tree->sort_potential[i] += pi_acc;
        tree->sort_fx[i]        += fxi_acc;
        tree->sort_fy[i]        += fyi_acc;
    }
}

/*
 * P2P_S：節點內部粒子的自交互（self interaction）。
 *   對節點內所有粒子對 (i, j) where i < j，利用對稱性同時更新兩端。
 *   self interaction 本質就是對稱的，is_symmetric 參數保留只是為了介面一致。
 */
void P2P_S(const HilbertTree *tree, int node_id)
{
    int s = tree->node_start[node_id];
    int e = tree->node_end[node_id];
    if (s >= e) return; /* 0 或 1 顆粒子無自交互可算 */

    for (int i = s; i <= e; i++) {
        double xi = tree->sort_x[i];
        double yi = tree->sort_y[i];
        double mi = tree->sort_mass[i];

        for (int j = i + 1; j <= e; j++) {
            double dx = xi - tree->sort_x[j];
            double dy = yi - tree->sort_y[j];
            double r2 = dx * dx + dy * dy;
            if (r2 < 1e-15) r2 = 1e-15;

            double mj    = tree->sort_mass[j];
            double pot   = 0.5 * log(r2);
            double invr2 = 1.0 / r2;
            double fxij  = dx * invr2;
            double fyij  = dy * invr2;

            tree->sort_potential[i] += G_CONST * mj * pot;
            tree->sort_potential[j] += G_CONST * mi * pot;
            tree->sort_fx[i] -= G_CONST * mj * fxij;
            tree->sort_fy[i] -= G_CONST * mj * fyij;
            tree->sort_fx[j] += G_CONST * mi * fxij;
            tree->sort_fy[j] += G_CONST * mi * fyij;
        }
    }
}
