#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "quadtree.h"

/* ================================================================
 *  Hilbert 曲線工具函式
 * ================================================================ */

static void hilbert_rot(int n, int *x, int *y, int rx, int ry)
{
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        int t = *x;
        *x = *y;
        *y = t;
    }
}

/*
 * 將 (x, y) 格點座標轉換為 Hilbert index。
 * n = 2^order（格點邊長，必須是 2 的冪）。
 */
uint64_t hilbert_xy2d(int n, int x, int y)
{
    uint64_t d = 0;
    for (int s = n >> 1; s > 0; s >>= 1) {
        int rx = (x & s) ? 1 : 0;
        int ry = (y & s) ? 1 : 0;
        d += (uint64_t)s * s * ((3 * rx) ^ ry);
        hilbert_rot(s, &x, &y, rx, ry);
    }
    return d;
}

/*
 * 將 Hilbert index d 反轉為 (x, y) 格點座標。
 * n = 2^order。
 */
void hilbert_d2xy(int n, uint64_t d, int *x, int *y)
{
    *x = *y = 0;
    for (int s = 1; s < n; s <<= 1) {
        int rx = (d / 2) & 1;
        int ry = (d ^ rx) & 1;
        hilbert_rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        d >>= 2;
    }
}

/* ================================================================
 *  qsort 比較函式（透過全域指標存取 hilbert_keys）
 * ================================================================ */

static const uint64_t *g_keys;

static int cmp_by_hilbert(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    if (g_keys[ia] < g_keys[ib]) return -1;
    if (g_keys[ia] > g_keys[ib]) return  1;
    return 0;
}

/* ================================================================
 *  二分搜尋：在 sort_indices[lo..hi] 中找第一個
 *  hilbert_keys[sort_indices[j]] >= target 的位置。
 *  若不存在則回傳 hi + 1。
 * ================================================================ */

static int lower_bound(const int *sort_indices, const uint64_t *keys,
                       int lo, int hi, uint64_t target)
{
    int left = lo, right = hi + 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (keys[sort_indices[mid]] < target)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

/* ================================================================
 *  hilbert_tree_build
 * ================================================================ */

HilbertTree *hilbert_tree_build(const Particle *particles, int n,
                                int max_per_leaf, int max_level)
{
    /* ---------- 1. 計算 bounding box ---------- */
    double xmin = DBL_MAX,  ymin = DBL_MAX;
    double xmax = -DBL_MAX, ymax = -DBL_MAX;
    for (int i = 0; i < n; i++) {
        if (particles[i].x < xmin) xmin = particles[i].x;
        if (particles[i].x > xmax) xmax = particles[i].x;
        if (particles[i].y < ymin) ymin = particles[i].y;
        if (particles[i].y > ymax) ymax = particles[i].y;
    }
    double dx = xmax - xmin, dy = ymax - ymin;
    double domain = (dx > dy ? dx : dy) * 1.01;
    if (domain < 1e-15) domain = 1.0;
    double ox = (xmin + xmax - domain) * 0.5;
    double oy = (ymin + ymax - domain) * 0.5;

    /* ---------- 2. 自動決定 max_level ---------- */
    if (max_level <= 0) {
        max_level = (int)ceil(log((double)n / max_per_leaf) / log(4.0));
        if (max_level < 1) max_level = 1;
    }
    if (max_level > MAX_LEVEL) max_level = MAX_LEVEL;

    int grid_n = 1 << max_level;  /* 2^max_level */

    /* ---------- 3. 計算每個粒子的 Hilbert index ---------- */
    uint64_t *keys = (uint64_t *)malloc(n * sizeof(uint64_t));
    int *sort_idx  = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        double nx = (particles[i].x - ox) / domain;
        double ny = (particles[i].y - oy) / domain;
        int gx = (int)(nx * grid_n);
        int gy = (int)(ny * grid_n);
        if (gx >= grid_n) gx = grid_n - 1;
        if (gy >= grid_n) gy = grid_n - 1;
        if (gx < 0) gx = 0;
        if (gy < 0) gy = 0;
        keys[i] = hilbert_xy2d(grid_n, gx, gy);
        sort_idx[i] = i;
    }

    /* ---------- 4. 依 Hilbert index 排序 ---------- */
    g_keys = keys;
    qsort(sort_idx, n, sizeof(int), cmp_by_hilbert);

    /* ---------- 5. 配置節點陣列 ---------- */
    /* total_nodes = sum_{l=0}^{max_level} 4^l = (4^(max_level+1) - 1) / 3 */
    int total_nodes = 0;
    {
        long long pw = 1;
        for (int l = 0; l <= max_level; l++) {
            total_nodes += (int)pw;
            pw *= 4;
        }
    }

    int    *nstart = (int *)malloc(total_nodes * sizeof(int));
    int    *nend   = (int *)malloc(total_nodes * sizeof(int));
    double *ncx    = (double *)malloc(total_nodes * sizeof(double));
    double *ncy    = (double *)malloc(total_nodes * sizeof(double));
    double *nsz    = (double *)malloc(total_nodes * sizeof(double));

    /* 預設所有節點為空 (start > end) */
    for (int i = 0; i < total_nodes; i++) {
        nstart[i] = 0;
        nend[i]   = -1;
    }

    /* leaf 收集用動態陣列 */
    int  leaf_cap = (n > 64) ? n : 64;
    int *leaves   = (int *)malloc(leaf_cap * sizeof(int));
    int  num_leaves = 0;

    /* ---------- 6. 根節點 ---------- */
    nstart[0] = 0;
    nend[0]   = n - 1;

    /* ---------- 7. 逐層建樹：二分搜尋找分割點 ---------- */
    int base_cur = 0;   /* base(level) = 當前層起始索引 */
    int count_cur = 1;  /* 當前層節點數 = 4^level */

    for (int lv = 0; lv < max_level; lv++) {
        int base_next = base_cur + count_cur;
        /* stride = 每個子節點涵蓋的 Hilbert range 大小 */
        uint64_t cells_per_child = 1;
        for (int k = 0; k < max_level - lv - 1; k++)
            cells_per_child *= 4;

        for (int i = 0; i < count_cur; i++) {
            int node_id = base_cur + i;
            int s = nstart[node_id];
            int e = nend[node_id];
            int np = e - s + 1;

            if (np <= 0) continue;          /* 空節點，跳過 */

            //先暫時棄用，算到max。
            // if (np <= max_per_leaf) {       /* 粒子數夠少 → leaf，不再分割 */
            //     leaves[num_leaves++] = node_id;
            //     continue;
            // }

            /* 執行分割 */
            uint64_t h_base = (uint64_t)i * cells_per_child * 4;

            int cursor = s;
            for (int c = 0; c < 4; c++) {
                int child_id = base_next + 4 * i + c;
                uint64_t h_upper = h_base + (uint64_t)(c + 1) * cells_per_child;

                int split;
                if (cursor > e) {
                    split = cursor;
                } else {
                    split = lower_bound(sort_idx, keys, cursor, e, h_upper);
                }

                nstart[child_id] = cursor;
                nend[child_id]   = split - 1;
                cursor = split;
            }
        }

        base_cur = base_next;
        count_cur *= 4;
    }

    /* 最深層：所有非空節點都是 leaf */
    for (int i = 0; i < count_cur; i++) {
        int node_id = base_cur + i;
        if (nstart[node_id] <= nend[node_id])
            leaves[num_leaves++] = node_id;
    }

    leaves = (int *)realloc(leaves, num_leaves * sizeof(int));

    /* ---------- 8. 計算每個節點的幾何資訊 ---------- */
    {
        int base = 0;
        int cnt  = 1;
        for (int lv = 0; lv <= max_level; lv++) {
            double cell_size = domain / (1 << lv);
            int grid_at_lv = 1 << lv;
            for (int i = 0; i < cnt; i++) {
                int node_id = base + i;
                /* 節點 i 在 level lv 的序號 → 用 d2xy 反推格點座標 */
                int gx, gy;
                hilbert_d2xy(grid_at_lv, (uint64_t)i, &gx, &gy);
                ncx[node_id] = ox + (gx + 0.5) * cell_size;
                ncy[node_id] = oy + (gy + 0.5) * cell_size;
                nsz[node_id] = cell_size;
            }
            base += cnt;
            cnt  *= 4;
        }
    }

    /* ---------- 9. 封裝回傳 ---------- */
    HilbertTree *tree = (HilbertTree *)malloc(sizeof(HilbertTree));
    tree->sort_indices  = sort_idx;
    tree->node_start    = nstart;
    tree->node_end      = nend;
    tree->hilbert_keys  = keys;
    tree->node_cx       = ncx;
    tree->node_cy       = ncy;
    tree->node_size     = nsz;
    tree->leaf_indices  = leaves;
    tree->num_leaves    = num_leaves;
    tree->origin_x      = ox;
    tree->origin_y      = oy;
    tree->domain_size   = domain;
    tree->max_level     = max_level;
    tree->total_nodes   = total_nodes;
    tree->n_particles   = n;

    return tree;
}

/* ================================================================
 *  hilbert_tree_free
 * ================================================================ */

void hilbert_tree_free(HilbertTree *tree)
{
    if (!tree) return;
    free(tree->sort_indices);
    free(tree->node_start);
    free(tree->node_end);
    free(tree->hilbert_keys);
    free(tree->node_cx);
    free(tree->node_cy);
    free(tree->node_size);
    free(tree->leaf_indices);
    free(tree);
}
