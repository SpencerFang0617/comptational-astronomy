#include "fmm.h"
#include "operators.h"

/*
 * 用 Pascal's rule 迭代填出 C(n, k)，n, k = 0 .. 2*MAX_P - 1。
 *   C(n, 0)   = 1
 *   C(n, n)   = 1
 *   C(n, k)   = C(n-1, k-1) + C(n-1, k),  0 < k < n
 *   C(n, k)   = 0,  k > n
 * 索引方式：binom[n][k]。M2M / M2L / L2L 都會用到。
 */
void precompute_binomial(double binom[2 * MAX_P][2 * MAX_P])
{
    const int N = 2 * MAX_P;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            binom[i][j] = 0.0;

    binom[0][0] = 1.0;
    for (int n = 1; n < N; n++) {
        binom[n][0] = 1.0;
        for (int k = 1; k <= n; k++)
            binom[n][k] = binom[n - 1][k - 1] + binom[n - 1][k];
    }
}
