#ifndef FMM_H
#define FMM_H

#include <complex.h>
#include <stdint.h>
#include "fmm.h"

#define FMM_ORDER    12
#define MAX_P        (FMM_ORDER + 1)
#define MAX_CHILDREN 4
#define MAX_LEVEL    20
#define MAX_PER_LEAF 64
#define G_CONST      1.0




/*
 * imput : 
 * 內部：for (j = tree->node_start[leaf_id]; j <= tree->node_end[leaf_id]; j++) { ... }
 *   a[0] = sum(m_i)
 *   a[k] = sum(m_i * (z_i - z_c)^k),  k = 1..p
 *  Finished
 */
double get_spherical_para(const Particle *particles, );

void get_Legendre();

void get_SpheriHarmo();


#endif