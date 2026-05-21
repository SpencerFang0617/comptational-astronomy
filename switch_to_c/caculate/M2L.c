#include "fmm.h"
#include "operators.h"

void M2L(const HilbertTree *tree, int node_id,int source_id,
    const double complex *multipole, double complex *local_exp,
    const double binom[2*MAX_P][2*MAX_P])
{
    double complex d = (tree->node_cx[source_id] - tree->node_cx[node_id])
                     + I * (tree->node_cy[source_id] - tree->node_cy[node_id]);
    double complex inv_d = 1 / d;
   
    /*將multipole的資料搬遷到a*/
    double complex *a = (double complex *)malloc(sizeof(double complex) * MAX_P);
    if (a != NULL) {
        memcpy(a, &multipole[source_id * MAX_P], sizeof(double complex) * MAX_P);
    }
    
    /* 0 階含 log(d) 項：d 是複數，必須用 clog 而不是 log，
     * 否則 log() 會把 complex 隱式轉成 double（取實部），
     * 當 creal(-d) < 0 時就會回傳 NaN。 */
    local_exp[node_id * MAX_P] += a[0] * clog(-d);

    for (int k = 1; k < MAX_P; k++){
        local_exp[node_id * MAX_P] += a[k] * inv_d * (k % 2 == 0 ? 1 : -1);
        inv_d /= d;
    }

    //高階項使用二項式卷積
    inv_d = 1 / d;
    for (int k = 1; k < MAX_P; k++){
        local_exp[node_id * MAX_P + k] += -a[0] * inv_d / k;
        double complex inv_d_pow = inv_d;
        inv_d /= d;
        for (int j = 1; j <= MAX_P - 1; j++){
            inv_d_pow /= d;
            local_exp[node_id * MAX_P + k] += binom[k + j - 1][j - 1] * a[j] * inv_d_pow * (j % 2 == 0 ? 1 : -1);
        }
    }
    free(a);
}