#include "fmm.h"
#include "operators.h"

void M2M(const HilbertTree *tree, int parent_id, int child_id, double complex *multipole,
    const double binom[2 * MAX_P][2 * MAX_P])
{
    
    double complex d = (tree->node_cx[child_id] - tree->node_cx[parent_id])
                     + I * (tree->node_cy[child_id] - tree->node_cy[parent_id]);
    double complex z_pow = 1;

    multipole[parent_id * MAX_P] += multipole[child_id * MAX_P];

    for (int k = 1; k < MAX_P; k++){
        z_pow *= d;
        multipole[parent_id * MAX_P + k] -= binom[k][0] * multipole[child_id * MAX_P] * z_pow / (double)k;

        /* 公式: parent.a[k] += sum_{j=1}^{k} C(k-1, j-1) * child.a[j] * d^(k-j)
         * 注意 j 必須從 1 開始，否則 binom[k-1][-1] 會越界讀取記憶體（UB）。 */
        double complex d_pow = 1;
        for (int j = k; j >= 1; j--){
            multipole[parent_id * MAX_P + k] += binom[k-1][j-1] * multipole[child_id * MAX_P + j] * d_pow;
            d_pow *= d;
        }
    }
}