#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// =====================================================================
// 1. 參數與結構定義
// =====================================================================

#define P_TERMS 4  
#define P_TOTAL ((P_TERMS + 1) * (P_TERMS + 2) / 2)  
#define MAX_PARTICLES 10   // finest box 粒子數上限

typedef struct {
    double real;
    double imag;
} Complex;

typedef struct {
    double x, y, z;
    double charge;
    double potential;
} Particle;

typedef struct Box {
    double cx, cy, cz;      
    double size;            
    int level;              
    int is_leaf;            
    
    int* particle_indices;  
    int num_particles;
    int capacity;
    
    struct Box* parent;     
    struct Box* children[8];
    
    Complex multipole[P_TOTAL]; 
    Complex local[P_TOTAL];     
} Box;

// --- 函數原型宣告 ---
Box* create_box(double cx, double cy, double cz, double size, int level, Box* parent);
void subdivide(Box* box);
void build_uniform_tree(Box* box,int MAX_LEVEL);
void insert_particle(Box* box, int p_idx, Particle* particles);
Box* find_box(Box* current, double x, double y, double z, int target_level);
int is_neighbor(Box* a, Box* b);
Box* get_neighbor(Box* box, int neighbor_index, Box* root);

// --- 複數運算 ---
Complex make_complex(double r, double i) {
    Complex c = {r, i};
    return c;
}

Complex c_add(Complex a, Complex b) {
    return make_complex(a.real + b.real, a.imag + b.imag);
}

Complex c_mul_real(Complex a, double r) {
    return make_complex(a.real * r, a.imag * r);
}

Complex c_mul_c(Complex a, Complex b){
    return(make_complex(a.real * b.real - a.imag - b.imag, a.imag * b.real + a.real * b.imag));
}

// =====================================================================
// 2. 空間搜尋與均勻樹管理
// =====================================================================

Box* create_box(double cx, double cy, double cz, double size, int level, Box* parent) {
    Box* box = (Box*)malloc(sizeof(Box));
    box->cx = cx; box->cy = cy; box->cz = cz;
    box->size = size;
    box->level = level;
    box->parent = parent;
    
    box->num_particles = 0;
    box->capacity = 10; // 初始容量
    box->particle_indices = (int*)malloc(sizeof(int) * box->capacity);
    box->is_leaf = 1;
    
    for (int i = 0; i < 8; i++) box->children[i] = NULL;
    memset(box->multipole, 0, sizeof(Complex) * P_TOTAL);
    memset(box->local, 0, sizeof(Complex) * P_TOTAL);
    return box;
}

void subdivide(Box* box) {
    box->is_leaf = 0;
    // 內部節點不儲存粒子，釋放記憶體節省空間
    free(box->particle_indices);
    box->particle_indices = NULL;
    
    double new_size = box->size / 2.0;
    for (int i = 0; i < 8; i++) {
        double ox = ((i & 1) ? 0.25 : -0.25) * box->size;
        double oy = (((i >> 1) & 1) ? 0.25 : -0.25) * box->size;
        double oz = (((i >> 2) & 1) ? 0.25 : -0.25) * box->size;
        box->children[i] = create_box(box->cx + ox, box->cy + oy, box->cz + oz, new_size, box->level + 1, box);
    }
}

// 一開始就強迫切分到 MAX_LEVEL 的均勻樹構建函數
void build_uniform_tree(Box* box,int MAX_LEVEL) {
    if (box->level >= MAX_LEVEL) return;
    subdivide(box);
    for (int i = 0; i < 8; i++) {
        build_uniform_tree(box->children[i],MAX_LEVEL);
    }
}

void insert_particle(Box* box, int p_idx, Particle* particles) {
    if (box->is_leaf) {
        if (box->num_particles >= box->capacity) {
            box->capacity *= 2;
            box->particle_indices = (int*)realloc(box->particle_indices, sizeof(int) * box->capacity);
        }
        box->particle_indices[box->num_particles++] = p_idx;
    } else {
        int idx = 0;
        if (particles[p_idx].x > box->cx) idx |= 1;
        if (particles[p_idx].y > box->cy) idx |= 2;
        if (particles[p_idx].z > box->cz) idx |= 4;
        insert_particle(box->children[idx], p_idx, particles);
    }
}

Box* find_box(Box* current, double x, double y, double z, int target_level) {
    if (!current || current->level > target_level) return NULL;

    double half = current->size / 2.0;
    if (x < current->cx - half - 1e-9 || x > current->cx + half + 1e-9 ||
        y < current->cy - half - 1e-9 || y > current->cy + half + 1e-9 ||
        z < current->cz - half - 1e-9 || z > current->cz + half + 1e-9) {
        return NULL;
    }

    if (current->level == target_level || current->is_leaf) return current;

    int idx = 0;
    if (x > current->cx) idx |= 1;
    if (y > current->cy) idx |= 2;
    if (z > current->cz) idx |= 4;
    return find_box(current->children[idx], x, y, z, target_level);
}

int is_neighbor(Box* a, Box* b) {
    if (!a || !b) return 0;
    double dx = fabs(a->cx - b->cx);
    double dy = fabs(a->cy - b->cy);
    double dz = fabs(a->cz - b->cz);
    // 均勻網格中同一層的盒子 size 必定相同
    double threshold = a->size + 1e-9;
    return (dx < threshold && dy < threshold && dz < threshold);
}

Box* get_neighbor(Box* box, int neighbor_index, Box* root) {
    int dx = (neighbor_index % 3) - 1;
    int dy = ((neighbor_index / 3) % 3) - 1;
    int dz = (neighbor_index / 9) - 1;
    double nx = box->cx + dx * box->size;
    double ny = box->cy + dy * box->size;
    double nz = box->cz + dz * box->size;
    
    Box* neighbor = find_box(root, nx, ny, nz, box->level);
    if (neighbor != NULL && is_neighbor(box, neighbor)) {
        return neighbor;
    }
    return NULL;
}

// convenient functions
void init_const_array(double (*pAlm)[2*P_TERMS+1][2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1][2*P_TERMS+1]){
    double facto[4*P_TERMS + 1];
    facto[0] = 1;
    for(int i=1;i<4*P_TERMS+1;i++) facto[i] = facto[i-1] * (double)i;

    for(int l=0;l<2*P_TERMS+1;l++){
        for(int m=0;m<l+1;m++){
            (*pNlm)[l][m] =sqrt(facto[l-m] / facto[l+m]);
            (*pAlm)[l][m] =((l%2) ? -1.0:1.0) /sqrt(facto[l-m] * facto[l+m]); 
        }
    }
    return;
}


void get_sph_num(double x1, double x2, double y1, double y2, double z1, double z2, double (*pds)[3] ,double* pr, double* psintheta, double* pcostheta, Complex* pe_iphi){
    (*pds)[0] = x1 - x2;
    (*pds)[1] = y1 - y2;
    (*pds)[2] = z1 - z2;
    double dxy = pow((*pds)[0], 2) + pow((*pds)[1], 2);
    *pr = dxy + pow((*pds)[2], 2);
    *pcostheta = (*pds)[2] / *pr;
    *psintheta = dxy / *pr;
    pe_iphi->real = *pds[0] / pow(dxy, 2);
    pe_iphi->imag = *pds[1] / pow(dxy, 2);
    return;
}

void get_P(int Pscale, double sintheta, double costheta, double (*pPlm)[Pscale]){
    // for L2M    Pscale = 2*m + 1
    // for others Pscale = m + 1
    (pPlm)[0][0] = 1.0;
    for(int m=0;m<Pscale-1;m++){
        (pPlm)[m+1][m+1] = -(2*m + 1) * sintheta * (pPlm)[m][m];
        (pPlm)[m+1][m]   =  (2*m + 1) * costheta * (pPlm)[m][m];
    }
    for(int l=2;l<=Pscale-1;l++){
        for(int m=0;m<l;m++){
            (pPlm)[l][m] = (double)(2*l - 1) / (l - m) * costheta * (pPlm)[l-1][m] 
                          - (double)(l + m - 1) / (l - m) * (pPlm)[l-2][m];
        }
    }
    return;
}

void get_Y(int Yscale, Complex e_iphi, double (*pPlm)[Yscale], Complex (*pYlm)[Yscale], double (*pNlm)[2*P_TERMS+1]){
    for(int l=0;l<Yscale;l++){
        Complex exp_now = make_complex(1, 0);
        for(int m=0;m<=l;m++){
            (pYlm)[l][m] = c_mul_real(exp_now, (pNlm)[l][m] * (pPlm)[l][m]);
            exp_now = c_mul_c(exp_now, e_iphi); // exp(i*m*phi)
        }
    }
    return;
}

// =====================================================================
// 3. FMM 算子 (Operators)
// =====================================================================
// (這部分物理公式維持你修正後的正確版)

void p2m(Box* box, Particle* particles, double (*pNlm)[2*P_TERMS+1]) {
    if (!box->is_leaf) return;
    Complex Mlm[P_TERMS+1][2*P_TERMS+1];
    for(int i=0;i<=P_TERMS;i++) for(int j=0;j<=P_TERMS;j++) Mlm[i][j]=0;

    for (int i = 0; i < box->num_particles; i++){
        Particle* p = &particles[box->particle_indices[i]];
        double ds[3], r, sintheta, costheta, Plm[P_TERMS+1][P_TERMS+1];
        Complex e_iphi, Ylm[P_TERMS+1][P_TERMS+1];

        get_r_related(p->x, box->cx, p->y, box->cy, p->z, box->cz, 
                      ds, &r, &sintheta, &costheta, &e_iphi);
        get_P(P_TERMS + 1, sintheta, costheta, Plm);
        get_Y(P_TERMS + 1, e_iphi, Plm, Ylm, pNlm);
        double r_now = 1;
        for(int l=0;l<P_TERMS+1;l++){
            double coeff_indep_m = particles->charge ** r_now;
            for(int m=-l;m<=l;m++){
                if(m>=0) Mlm[l][P_TERMS+m] += r_now * Ylm[l][m];
                else Mlm[l][P_TERMS+m] += r_now * make_complex(Ylm[l][-m].real, -Ylm[l][-m].imag); // conj of Y
            }
            r_now *= r;
        }
        box->multipole = Mlm;
    }
}

void m2m(Box* parent, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    memset(parent->multipole, 0, sizeof(Complex) * P_TOTAL);
    Complex i_power[4] = {make_complex(0, 1), make_complex(-1, 0), make_complex(0, -1), make_complex(1, 0)};
    for(int i=0;i<8;i++){
        double ds[3], r, sintheta, costheta, Plm[P_TERMS+1][P_TERMS+1];
        Complex e_iphi, Ylm[P_TERMS+1][P_TERMS+1];
        Box *child = parent->children[i];
        get_sph_num(child->cx, parent->cx, child->cy, parent->cy, child->cz, parent->cz, ds, &r, &sintheta, &costheta, &e_iphi);
        get_P(P_TERMS+1, sintheta, costheta, Plm);
        get_Y(P_TERMS+1, e_iphi, Plm, Ylm, pNlm);

        Complex (*Olm)[2*P_TERMS+1] = child->multipole;
        for(int j=0;j<=P_TERMS;j++){
            for(int k=-j;k<=j;k++){
                Complex add_num = make_complex(0, 0);
                double r_now = 1;
                for(int n=0;n<=j;n++){
                    for(int m=-n;m<=n;m++){
                        if(abs(k-m) > j-n) continue;
                        Complex flag;
                        if(m>=0) flag = make_complex(Ylm[n][m].real, -Ylm[n][m].imag);
                        else flag = Ylm[n][-m];
                        

                        add_num = c_add(add_num,
                                    c_mul_real(
                                        c_mul_c(
                                            c_mul_c(Olm[j-n][P_TERMS+k-m], i_power[(((abs(k)-abs(m)-abs(k-m))%4)+4)%4]),
                                            flag),
                                        (*pAlm)[n][abs(m)] * (*pAlm)[j-n][abs(k-m)] * r_now));
                    }
                    r_now *= r;
                }
                parent->multipole[j][P_TERMS+k] = c_add(parent->multipole[j][P_TERMS+k],
                                                        c_mul_real(add_num, 1.0 / (*pAlm)[j][abs(k)]));
            }
        }
    }
    return;
}

void m2l(Box* target, Box* source, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    double ds[3], r, sintheta, costheta, Plm[2*P_TERMS+1][2*P_TERMS+1];
    Complex e_iphi, Ylm[2*P_TERMS+1][2*P_TERMS+1];
    Complex i_power[4] = {make_complex(0, 1), make_complex(-1, 0), make_complex(0, -1), make_complex(1, 0)};
    get_sph_num(source->cx, target->cx, source->cy, target->cy, source->cz, target->cz, ds, &r, &sintheta, &costheta, &e_iphi);
    get_P(2*P_TERMS+1, sintheta, costheta, Plm);
    get_Y(2*P_TERMS+1, e_iphi, Plm, Ylm, pNlm);

    Complex (*Olm)[2*P_TERMS+1] = source->multipole;
    for(int j=0;j<=P_TERMS;j++){
        for(int k=-j;k<=j;k++){
            double m1_now = 1, rnow = pow(r, j+1);
            Complex add_num = make_complex(0, 0);
            for(int n=0;n<=P_TERMS;n++){
                for(int m=-n;m<=n;m++){
                    Complex flag;
                    if(m-k<0) flag = make_complex(Ylm[j+n][-m+k].real, -Ylm[j+n][-m+k].imag);
                    else flag = Ylm[j+n][m-k];

                    add_num = c_add(add_num, 
                                c_mul_real(
                                    c_mul_c(
                                        c_mul_c(Olm[n][P_TERMS+m], i_power[((abs(k-m)-abs(k)-abs(m))%4+4)%4]), 
                                                flag), 
                                    pAlm[n][abs(m)] * pAlm[j][abs(k)]/ (m1_now * rnow * pAlm[j+n][abs(m-k)])));
                }
                m1_now *= -1;
                rnow *= r;
            }
            target->local[j][P_TERMS+k] = c_add(target->local[j][P_TERMS+k], add_num);
        }
    }
    return;
}

void l2l(Box* parent) {
    if (parent->is_leaf) return;
    for (int i = 0; i < 8; i++) {
        Box* child = parent->children[i];
        if (!child) continue;
        double dist = sqrt(pow(child->cx - parent->cx, 2) + pow(child->cy - parent->cy, 2) + pow(child->cz - parent->cz, 2));
        int idx = 0;
        for (int l = 0; l <= P_TERMS; l++) {
            for (int m = 0; m <= l; m++) {
                double weight = pow(dist, l);
                child->local[idx] = c_add(child->local[idx], c_mul_real(parent->local[idx], weight));
                idx++;
            }
        }
    }
}

void l2p(Box* box, Particle* particles) {
    if (!box->is_leaf) return;
    for (int i = 0; i < box->num_particles; i++) {
        Particle* p = &particles[box->particle_indices[i]];
        double r = sqrt(pow(p->x - box->cx, 2) + pow(p->y - box->cy, 2) + pow(p->z - box->cz, 2));
        int idx = 0;
        for (int l = 0; l <= P_TERMS; l++) {
            for (int m = 0; m <= l; m++) {
                p->potential += box->local[idx].real * pow(r, l);
                idx++;
            }
        }
    }
}

// =====================================================================
// 運算執行
// =====================================================================

void upward_pass(Box* box, Particle* particles) {
    if (box->is_leaf) {
        p2m(box, particles);
    } else {
        for (int i = 0; i < 8; i++) {
            if (box->children[i]) upward_pass(box->children[i], particles);
        }
        m2m(box);
    }
}

void compute_m2l(Box* target_box, Box* root) {
    Box* parent = target_box->parent;
    if (!parent) return;

    for (int i = 0; i < 27; i++) {
        Box* p_neighbor = get_neighbor(parent, i, root);
        if (!p_neighbor) continue;

        // 均勻網格中，p_neighbor 絕對不是葉子，直接去抓它的 8 個子盒子即可
        for (int j = 0; j < 8; j++) {
            Box* candidate = p_neighbor->children[j];
            if (candidate && !is_neighbor(target_box, candidate)) {
                m2l(target_box, candidate);
            }
        }
    }
}

void downward_pass(Box* box, Box* root) {
    compute_m2l(box, root);
    l2l(box); 
    
    if (!box->is_leaf) {
        for (int i = 0; i < 8; i++) {
            if (box->children[i]) downward_pass(box->children[i], root);
        }
    }
}

// 均勻網格不再需要遞迴撈取葉子，鄰居一定是跟你一樣大的葉子
void compute_near_field_uniform(Box* target_leaf, Box* neighbor_leaf, Particle* particles) {
    if (!neighbor_leaf || !neighbor_leaf->is_leaf) return;

    for (int i = 0; i < target_leaf->num_particles; i++) {
        for (int j = 0; j < neighbor_leaf->num_particles; j++) {
            int idx1 = target_leaf->particle_indices[i];
            int idx2 = neighbor_leaf->particle_indices[j];

            double dx = particles[idx1].x - particles[idx2].x;
            double dy = particles[idx1].y - particles[idx2].y;
            double dz = particles[idx1].z - particles[idx2].z;
            double r = sqrt(dx*dx + dy*dy + dz*dz);

            if (r > 1e-10) {
                // 雙向加總
                particles[idx1].potential += particles[idx2].charge / r;
                particles[idx2].potential += particles[idx1].charge / r;
            }
        }
    }
}

void evaluate(Box* box, Box* root, Particle* particles) {
    if (box->is_leaf) {
        l2p(box, particles); 

        // (a) 盒內粒子對（自身 vs 自身）
        for (int i = 0; i < box->num_particles; i++) {
            for (int j = i + 1; j < box->num_particles; j++) {
                int idx1 = box->particle_indices[i];
                int idx2 = box->particle_indices[j];
                double dx = particles[idx1].x - particles[idx2].x;
                double dy = particles[idx1].y - particles[idx2].y;
                double dz = particles[idx1].z - particles[idx2].z;
                double r = sqrt(dx*dx + dy*dy + dz*dz);
                if (r > 1e-10) {
                    particles[idx1].potential += particles[idx2].charge / r;
                    particles[idx2].potential += particles[idx1].charge / r;
                }
            }
        }

        // (b) 與鄰居盒子的粒子對：只取 n > 13 的方向，搭配上面雙向加總避免重複
        for (int n = 14; n < 27; n++) {
            Box* neighbor = get_neighbor(box, n, root);
            if (!neighbor) continue;
            compute_near_field_uniform(box, neighbor, particles);
        }
    } else {
        for (int i = 0; i < 8; i++) {
            if (box->children[i]) evaluate(box->children[i], root, particles);
        }
    }
}

void free_tree(Box* box) {
    if (!box) return;
    if (!box->is_leaf) {
        for (int i = 0; i < 8; i++) free_tree(box->children[i]);
    }
    if (box->particle_indices) free(box->particle_indices);
    free(box);
}

int main() {
    int N = 500;
    int MAX_LEVEL=(int)(log((double)N / MAX_PARTICLES) / log(8.0));
    double Nlm[2*P_TERMS+1][2*P_TERMS+1], Alm[2*P_TERMS+1][2*P_TERMS+1]; // some const array needed for P2M and etc...
    Particle* particles = (Particle*)malloc(sizeof(Particle) * N);

    srand(42);
    for (int i = 0; i < N; i++) {
        particles[i].x = (double)rand() / RAND_MAX;
        particles[i].y = (double)rand() / RAND_MAX;
        particles[i].z = (double)rand() / RAND_MAX;
        particles[i].charge = 1.0;
        particles[i].potential = 0.0;
    }
    // build Nlm,Alm
    init_const_array(&Alm, &Nlm);

    // 建立根節點
    Box* root = create_box(0.5, 0.5, 0.5, 1.0, 0, NULL);
    build_uniform_tree(root,MAX_LEVEL);
    for (int i = 0; i < N; i++) insert_particle(root, i, particles);

    upward_pass(root, particles, &Alm, &Nlm);
    downward_pass(root, root, &Alm, &Nlm);
    evaluate(root, root, particles, &Alm, &Nlm);

    double direct_pot = 0;
    for (int i = 1; i < N; i++) {
        double r = sqrt(pow(particles[0].x-particles[i].x,2)+pow(particles[0].y-particles[i].y,2)+pow(particles[0].z-particles[i].z,2));
        if (r > 1e-10) direct_pot += particles[i].charge / r;
    }
    free_tree(root);
    free(particles);
    return 0;
}