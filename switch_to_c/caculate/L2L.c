#include "fmm.h"
#include "operators.h"

void L2L(const HilbertTree *tree, int parent_id, int child_id,
    double complex *local_exp,const double binom[2*MAX_P][2*MAX_P])
{
    double complex d = (tree->node_cx[child_id] - tree->node_cx[parent_id])
                     + I * (tree->node_cy[child_id] - tree->node_cy[parent_id]);;
    //fprintf(fp_L2L, "%d %d ", parent_id, child_id);
    //fprintf(fp_L2L, "%.*f ", 15, cimag(d));
    //fprintf(fp_L2L, "%.*f ", 15, creal(d));
    //fprintf(fp_L2L, " ");
    for (int k = 0; k < MAX_P; k++){
        double complex d_pow = 1;
        for (int j = k; j < MAX_P; j++){
#ifdef DEBUG
            {
                double b = binom[j][k];
                if (!isfinite(b)) {
                    fprintf(stderr, "binom[%d][%d] = %g\n", j, k, b);
                    exit(1);
                }
                double complex le = local_exp[parent_id * MAX_P + j];
                if (!isfinite(creal(le)) || !isfinite(cimag(le))) {
                    fprintf(stderr, "local_exp[%d][%d] re=%g im=%g\n",
                            parent_id, j, creal(le), cimag(le));
                    exit(1);
                }
                if (!isfinite(creal(d_pow)) || !isfinite(cimag(d_pow))) {
                    fprintf(stderr, "d_pow re=%g im=%g\n", creal(d_pow), cimag(d_pow));
                    exit(1);
                }
            }
#endif
            local_exp[child_id * MAX_P + k] += binom[j][k] * local_exp[parent_id * MAX_P + j] * d_pow;
            d_pow *= d;
            //fprintf(fp_L2L, "%.*f ", 15, cimag(d_pow));
            //fprintf(fp_L2L, "%.*f ", 15, creal(d_pow));
            //fprintf(fp_L2L, " ");
        }
        // 輸出完整小數位數
        //fprintf(fp_L2L, "%.*f ", 15, cimag(local_exp[child_id * MAX_P + k]));
        //fprintf(fp_L2L, "%.*f ", 15, creal(local_exp[child_id * MAX_P + k]));
    }
    //fprintf(fp_L2L, "\n");
}

/* Local-to-Local：父節點局部展開平移到子節點中心。
 *   d = z_child_center - z_parent_center
 *   child.b[k] += sum_{j=k}^{p} C(j,k) * parent.b[j] * d^(j-k)
 */