#include "fmm.h"
#include "operators.h"

void L2P(const HilbertTree *tree, int leaf_id, double complex *local_exp){
    int start = tree->node_start[leaf_id];
    int end = tree->node_end[leaf_id];
    if (start > end) return;

    for (int i = start; i <= end; i++){
        double complex dz = (tree->sort_x[i] - tree->node_cx[leaf_id]) + I * (tree->sort_y[i] - tree->node_cy[leaf_id]);
        /* 計算勢能: Phi(z) = Re( sum( b_k * (z - z_c)^k ) )*/
        double complex pot = 0;
        double complex dz_pow = 1;
        for (int k = 0; k < MAX_P; k++){
            pot += local_exp[leaf_id * MAX_P + k] * dz_pow;
            dz_pow *= dz;
        }
        tree->sort_potential[i] += G_CONST * creal(pot);

        /* 計算受力 (勢能的負梯度): F = -conj( dPhi/dz ) */
        double complex force = 0;
        dz_pow = 1;
        for (int k = 1; k < MAX_P; k++){
            force += (double complex)k * local_exp[leaf_id * MAX_P + k] * dz_pow;
            dz_pow *= dz;
        }
        /* 因為 F = -conj(dPhi/dz)，所以 Fx = -Re(dPhi/dz)，Fy = +Im(dPhi/dz) */
        tree->sort_fx[i] -= G_CONST * creal(force);
        tree->sort_fy[i] += G_CONST * cimag(force);

    }
}
/*
 * Local-to-Particle：從局部展開係數計算葉節點每個粒子的勢能和力。
 *   phi = Re( sum_k b[k] * (z - z_c)^k )
 *   F   = -conj( dPhi/dz )
 *  Working
 */