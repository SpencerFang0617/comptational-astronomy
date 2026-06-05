#include "operators.h"

void P2M(const HilbertTree *tree, int leaf_id, double complex *multipole){


    
}
/*
 * Multipole-to-Multipole：子節點多極展開平移到父節點中心。
 *   d = z_child_center - z_parent_center
 *   parent.a[k] += sum_{j=0}^{k} C(k,j) * child.a[j] * d^(k-j)
 *  Finished
 */