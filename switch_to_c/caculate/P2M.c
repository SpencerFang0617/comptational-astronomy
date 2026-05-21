#include "fmm.h"
#include "operators.h"

void P2M(const HilbertTree *tree, int leaf_id, double complex *multipole){
    for (int j =  tree->node_start[leaf_id]; j <= tree->node_end[leaf_id]; j++){
        double complex z = (tree->sort_x[j] - tree->node_cx[leaf_id])
                         + I * (tree->sort_y[j] - tree->node_cy[leaf_id]);
        double m = tree->sort_mass[j];
        multipole[leaf_id * MAX_P ] += m;
        double complex z_pow = 1;
        for (int k = 1; k < MAX_P; k++){
            z_pow *= z;
            multipole[leaf_id * MAX_P + k] -= m * z_pow / (double)k;
        }
    }
}