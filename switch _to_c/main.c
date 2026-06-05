#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "fmm.h"
#include "quadtree.h"
#include "operators.h"

static void fmm_solve(Particle *particles, int n)
{
    double binom[2 * MAX_P][2 * MAX_P];
    precompute_binomial(binom);

    HilbertTree *tree = hilbert_tree_build(particles, n, MAX_PER_LEAF, MAX_LEVEL);
    /* P2M & M2M from deepest level upward */
    double complex *multipole = calloc(tree->total_nodes * MAX_P, sizeof(double complex));
    // 取第 node_id 個節點的第 k 個係數：multipole[node_id * MAX_P + k]
    
    for (int i = 0; i < tree->num_leaves; i++)
        P2M(particles[tree->sort_indices[tree->node_start[tree->leaf_indices[i]]]], multipole);

    /*從最深層開始，先判斷自己的M2M_result[i]是否為0，若為零則檢查子節點的M2M_result是否為0，若不為零則計算M2M，並將結果存入M2M_result[i]，如過M2M_result[i]不為零則跳過計算*/
    for (int lv = tree->max_level - 1; lv >= 0; lv--) {
        for (int idx = 0; idx < count[lv]; idx++) {
            int parent_id = base[lv] + idx;
            if (tree->node_start[parent_id] > tree->node_end[parent_id])
                continue;  // 空節點跳過
            for (int c = 0; c < 4; c++) {
                int child_id = base[lv+1] + 4 * idx + c;
                if (tree->node_start[child_id] > tree->node_end[child_id])
                    continue;  // 空子節點跳過
                M2M(tree, parent_id, child_id, multipole, binom);
                // ↑ 累加，不 break
            }
        }
    }
    /* ---- Downward pass: M2L + L2L，由頂到底 ---- */
    double complex *local_exp = calloc(tree->total_nodes * MAX_P, sizeof(double complex));

    /* 預計算每一層的 base 和 count */
    int base_arr[MAX_LEVEL + 1], count_arr[MAX_LEVEL + 1];
    base_arr[0] = 0;  count_arr[0] = 1;
    for (int lv = 1; lv <= tree->max_level; lv++) {
        base_arr[lv] = base_arr[lv-1] + count_arr[lv-1];
        count_arr[lv] = count_arr[lv-1] * 4;
    }

    /* 從 level 2 開始才有 interaction list */
    for (int lv = 2; lv <= tree->max_level; lv++) {
        int grid_lv      = 1 << lv;        // 2^lv
        int grid_parent  = 1 << (lv - 1);  // 2^(lv-1)

        for (int idx = 0; idx < count_arr[lv]; idx++) {
            int node_id = base_arr[lv] + idx;
            if (tree->node_start[node_id] > tree->node_end[node_id])
                continue;  // 空節點跳過

            /* --- L2L: 父 → 自己（父已在上一層處理完畢）--- */
            int parent_idx = idx / 4;
            int parent_id  = base_arr[lv-1] + parent_idx;
            L2L(tree, parent_id, node_id, local_exp, binom);

            /* --- M2L: 找 interaction list --- */
            int nx, ny;
            hilbert_d2xy(grid_lv, (uint64_t)idx, &nx, &ny);

            int px, py;
            hilbert_d2xy(grid_parent, (uint64_t)parent_idx, &px, &py);

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int npx = px + dx, npy = py + dy;
                    if (npx < 0 || npx >= grid_parent ||
                        npy < 0 || npy >= grid_parent)
                        continue;  // 超出邊界

                    uint64_t nb_parent_hid = hilbert_xy2d(grid_parent, npx, npy);
                    int nb_parent_idx = (int)nb_parent_hid;

                    for (int c = 0; c < 4; c++) {
                        int src_idx = 4 * nb_parent_idx + c;
                        int src_id  = base_arr[lv] + src_idx;

                        if (src_id == node_id) continue;  // 跳過自己
                        if (tree->node_start[src_id] > tree->node_end[src_id])
                            continue;  // 空節點跳過

                        /* 檢查是否為鄰居 */
                        int cx, cy;
                        hilbert_d2xy(grid_lv, (uint64_t)src_idx, &cx, &cy);

                        if (abs(cx - nx) <= 1 && abs(cy - ny) <= 1)
                            continue;  // 是鄰居 → 不屬於 interaction list

                        /* 是 interaction list 成員 → 做 M2L */
                        M2L(tree, node_id, src_id, multipole, local_exp, binom);
                    }
                }
            }
        }
    }

    /* ---- Final evaluation: leaves only ---- */
    for (int i = 0; i < tree->num_leaves; i++)
        L2P(particles[tree->sort_indices[tree->node_start[tree->leaf_indices[i]]]], particles);

    /*P2P evaluation for leaves*/
    for (int i = 0; i < tree->num_leaves; i++)
        P2P_S(particles[tree->sort_indices[tree->node_start[tree->leaf_indices[i]]]], particles[tree->sort_indices[tree->node_end[tree->leaf_indices[i]]]], binom);

        static int find_level(int node_id, const int *base_arr,
            const int *count_arr, int max_level)
            {
            for (int lv = max_level; lv >= 0; lv--)
            if (node_id >= base_arr[lv])
            return lv;
            return 0;
            }
    
    /*P2P evaluation for non-leaves*/
    /* 從 level 2 開始才有 interaction list */
    for (int i = 0; i < tree->num_leaves; i++) {
        int leaf_id = tree->leaf_indices[i];
        int lv  = find_level(leaf_id, base_arr, count_arr, tree->max_level);
        int idx = leaf_id - base_arr[lv];
        int grid_lv = 1 << lv;
        /* 自交互：leaf 內部粒子彼此作用 */
        P2P_self(tree, leaf_id, particles);
        /* 找同層鄰居 */
        int nx, ny;
        hilbert_d2xy(grid_lv, (uint64_t)idx, &nx, &ny);
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;
                int nbx = nx + dx, nby = ny + dy;
                if (nbx < 0 || nbx >= grid_lv || nby < 0 || nby >= grid_lv)
                    continue;
                int nb_idx = (int)hilbert_xy2d(grid_lv, nbx, nby);
                int nb_id  = base_arr[lv] + nb_idx;
                if (tree->node_start[nb_id] > tree->node_end[nb_id])
                    continue;
                /* 避免重複計算：只處理 nb_id > leaf_id 的一半 */
                if (nb_id <= leaf_id) continue;
                P2P_A(leaf_id, nb_id, binom);
            }
        }
    }
    
        
    
    /* cleanup */
    hilbert_tree_free(tree);
}

/* brute-force O(N^2) direct sum for verification */
static void direct_sum(Particle *particles, int n)
{
    for (int i = 0; i < n; i++) {
        particles[i].potential = 0.0;
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double r2 = dx * dx + dy * dy;
            if (r2 < 1e-15) r2 = 1e-15;

            double pot = 0.5 * log(r2);
            double inv_r2 = 1.0 / r2;

            particles[i].potential += G_CONST * particles[j].mass * pot;
            particles[j].potential += G_CONST * particles[i].mass * pot;

            double fx = dx * inv_r2;
            double fy = dy * inv_r2;
            particles[i].fx -= G_CONST * particles[j].mass * fx;
            particles[i].fy -= G_CONST * particles[j].mass * fy;
            particles[j].fx += G_CONST * particles[i].mass * fx;
            particles[j].fy += G_CONST * particles[i].mass * fy;
        }
    }
}

int main(int argc, char *argv[])
{
    int N = 1000;
    if (argc > 1) N = atoi(argv[1]);

    srand(42);
    Particle *particles = malloc(N * sizeof(Particle));
    for (int i = 0; i < N; i++) {
        particles[i].x = (double)rand() / RAND_MAX * 100.0;
        particles[i].y = (double)rand() / RAND_MAX * 100.0;
        particles[i].mass = (double)rand() / RAND_MAX + 0.5;
        particles[i].potential = 0.0;
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    printf("Running FMM with %d particles...\n", N);

    clock_t t0 = clock();
    fmm_solve(particles, N);
    clock_t t1 = clock();
    double fmm_time = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("FMM completed in %.4f sec\n", fmm_time);

    /* save FMM results for comparison */
    double *fmm_pot = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        fmm_pot[i] = particles[i].potential;

    /* run direct sum for verification (only for small N) */
    if (N <= 5000) {
        direct_sum(particles, N);

        double max_err = 0.0, max_ref = 0.0;
        for (int i = 0; i < N; i++) {
            double err = fabs(fmm_pot[i] - particles[i].potential);
            if (err > max_err) max_err = err;
            if (fabs(particles[i].potential) > max_ref)
                max_ref = fabs(particles[i].potential);
        }
        printf("Max relative potential error: %e\n",
               max_ref > 0 ? max_err / max_ref : max_err);
    }

    free(fmm_pot);
    free(particles);
    return 0;
}
