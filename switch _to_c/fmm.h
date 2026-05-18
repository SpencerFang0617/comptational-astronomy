#ifndef FMM_H
#define FMM_H

#include <complex.h>
#include <stdint.h>

#define FMM_ORDER    12
#define MAX_P        (FMM_ORDER + 1)
#define MAX_CHILDREN 4
#define MAX_LEVEL    20
#define MAX_PER_LEAF 64
#define G_CONST      1.0

typedef struct {
    double x, y;
    double mass;
    double potential;
    double fx, fy;
} Particle;

typedef struct QuadtreeNode {
    double cx, cy;
    double size;
    int level;
    int is_leaf;

    int *particle_indices;
    int num_particles;

    double complex multipole[MAX_P];
    double complex local_exp[MAX_P];

    struct QuadtreeNode *parent;
    struct QuadtreeNode *children[MAX_CHILDREN];

    struct QuadtreeNode **neighbors;
    int num_neighbors;
    struct QuadtreeNode **interaction_list;
    int num_interactions;
} QuadtreeNode;

typedef struct {
    int      *sort_indices;   /* 按 Hilbert 順序排列的粒子原始索引 */
    int      *node_start;     /* 每個節點在 sort_indices 中的起始位置 */
    int      *node_end;       /* 每個節點在 sort_indices 中的結束位置（含） */
    uint64_t *hilbert_keys;   /* 每個粒子的 Hilbert index（排序用，長度 N） */

    double   *node_cx;        /* 每個節點的中心 x */
    double   *node_cy;        /* 每個節點的中心 y */
    double   *node_size;      /* 每個節點的邊長 */

    int      *leaf_indices;    /* leaf 節點的 node id 陣列，可直接索引 node_start/node_end */
    int       num_leaves;     /* leaf 節點數量 */

    double    origin_x, origin_y; /* bounding box 左下角 */
    double    domain_size;        /* bounding box 邊長 */

    int       max_level;      /* 樹的最大深度 */
    int       total_nodes;    /* 節點總數 = (4^(max_level+1) - 1) / 3 */
    int       n_particles;    /* 粒子數 */
} HilbertTree;

#endif
