// FMFTRLOptimize.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <math.h>


// Dropped bias term (not performance critical)

double predict_single(int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
  double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, unsigned int D_fm, int threads) {
  
 
  unsigned int i, ii, k;
  double sign, zi, d, wi, wi2, wfmk, e = 0.0, e2 = 0.0;


  // for ii in prange(lenn,num_threads = threads) :
  for (ii = 0; ii < 100; ii++) {

    i = inds[ii];
    zi = z[i];

    if (zi < 0) {
      sign = -1.0;
    }
    else {
      sign = 1.0;
    }

    if (sign * zi > L1) {
      w[ii + 1] = wi = (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2);
      e += wi * vals[ii];
    }
    else
    {
      w[ii + 1] = 0.0;
    }

  }

  wi2 = 0.0;

  // for k in prange(D_fm, nogil = True, num_threads = threads) :
  for (k = 0; k < 100; k++) {

    wfmk = 0.0;
    //for ii in range(lenn) :

    // ASSUMED HOTSPOT
    for (int ii = 0; ii < 100; ii++) {

      // Consecutive read
      double this_v = vals[ii];

      // Consecutive read
      unsigned int idx = inds[ii];

      // In register calculations
      idx = idx * D_fm + k;

      // Gather load. Possibly AVX2: vgatherdpd
      d = z_fm[idx];
      
      // in register calculations
      d = d *this_v;

      // in register calculations
      wfmk = wfmk + d;

      // in register calculations
      wi2 += ( d * d);
    }
    e2 += (wfmk * wfmk);
    w_fm[k] = wfmk;
  }
  e2 = (e2 - wi2)* 0.5 *weight_fm;
  return e + e2;

}

// Dropping bias_term

void update_single(int* inds, double* vals, int lenn, double eConst, double ialpha, double* w, double* z, double* n, double alpha_fm, double L2_fm,
  double* w_fm, double* z_fm, double* n_fm, unsigned int D_fm, int threads) {

  unsigned int i, ii, k;

  double g, g2, ni, v, lr, e2 = eConst* eConst, reg, L2_fme = L2_fm / eConst;

  double *z_fmi;


  //for ii in prange(lenn, nogil = True, num_threads = threads) :
  // for ii in range(lenn)
  for (ii = 0; ii < 100; ii++) {

    // Read consecutive memory into register
    i = inds[ii];

    // Read consecutive memory into register
    v = vals[ii];

    // In register calculation
    g = eConst * v;

    // In register calculation
    g2 = g * g;


    // Read scattered memory
    ni = n[i];


    // Read consecutive memory +1
    double w_next = w[ii + 1];

    // In register calculation
    double temp = g - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w_next;

    // Scattered read and store
    z[i] += temp;

    // calc and scattered store
    n[i] = ni + g2;


    // In register calculation
    z_fmi = z_fm + i * D_fm;
    

    // Scattered read
    double this_nfm = n_fm[i];

    // In register calculation
    lr = g * alpha_fm / (sqrt(this_nfm) + 1.0);

    // In register calculation
    reg = v - L2_fme;


    /*
    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)

    */


    // D_fm about 20 - 200
    //for k in range(D_fm) {   
    
    // ASSUMED HOTSPOT
    for (k = 0; k < 100; k++) {

      // Read consecutive
      double w_fm_this = w_fm[k];

      // Read consecutive
      double z_fmi_this = z_fmi[k];

      // In register calculation
      double myTemp = z_fmi_this - lr * (w_fm_this - z_fmi_this * reg);

      // Consecutive store
      z_fmi[k] = myTemp;

    }
    n_fm[i] += e2;

  }
}


int main() {

  int* inds = new int[100];

  double* vals = new double[100];

  int lenn = 90;

  double e = 100.0;

  double ialpha = 0.01;

  double* w = new double[100];
  
  double* z = new double[100];
  
  double* n = new double[100];

  double alpha_fm = 0.1;

  double L2_fm = 0.1;

  double* w_fm = new double[100];

  double* z_fm = new double[100];

  double* n_fm = new double[100];



  unsigned int D_fm = 4;

  int threads = 4;

  double L1 = 0.1;
  double baL2 = 0.1;
  double beta = 0.3;
  double weight_fm = 0.1;

  predict_single(inds, vals, lenn, L1, baL2, ialpha, beta, w, z, n, w_fm, z_fm, n_fm, weight_fm, D_fm, threads);

  update_single(inds, vals, lenn, e, ialpha, w, z, n, alpha_fm, L2_fm, w_fm, z_fm, n_fm, D_fm, threads);

  

 



  delete[] inds;
  delete[] vals;
  delete[] w;
  delete[] z;
  delete[] n;

  delete[] w_fm;
  delete[] z_fm;
  delete[] n_fm;

  return 0;
}