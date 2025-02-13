#include "min_pimd.h"
#include "fix_pimd_langevin.h"
#include "modify.h"
#include "atom.h"
#include "update.h"
#include "error.h"
#include "math.h"

using namespace LAMMPS_NS;

MinPIMD::MinPIMD(LAMMPS *lmp) : Min(lmp) {}

void MinPIMD::init()
{
  Min::init();
  fix_pimd = static_cast<FixPIMDLangevin *>(modify->get_fix_by_id("pimd/langevin"));
  if (!fix_pimd) error->all(FLERR, "pimd/langevin fix not found");
}

void MinPIMD::force_clear()
{
  // Clear forces
  for (int i = 0; i < atom->nlocal; i++) {
    atom->f[i][0] = 0.0;
    atom->f[i][1] = 0.0;
    atom->f[i][2] = 0.0;
  }
}

void MinPIMD::force_compute()
{
  // Compute forces using PIMD fix
  fix_pimd->post_force(1);
}

void MinPIMD::run(int nsteps)
{
  init();
  
  // L-BFGS parameters
  const int m = 5; // memory parameter for L-BFGS
  const double tol = 1e-6; // tolerance for convergence
  const double alpha = 1.0; // step size
  
  // Allocate memory for L-BFGS
  double **s = new double*[m];
  double **y = new double*[m];
  double *rho = new double[m];
  double *alpha_vec = new double[m];
  for (int i = 0; i < m; i++) {
    s[i] = new double[3 * atom->nlocal];
    y[i] = new double[3 * atom->nlocal];
  }
  
  double *g = new double[3 * atom->nlocal];
  double *g_old = new double[3 * atom->nlocal];
  double *x_old = new double[3 * atom->nlocal];
  double *q = new double[3 * atom->nlocal];
  
  int k = 0;
  
  for (int step = 0; step < nsteps; step++) {
    force_clear();
    force_compute();
    
    // Copy current positions and forces
    for (int i = 0; i < atom->nlocal; i++) {
      x_old[3*i] = atom->x[i][0];
      x_old[3*i+1] = atom->x[i][1];
      x_old[3*i+2] = atom->x[i][2];
      g[3*i] = atom->f[i][0];
      g[3*i+1] = atom->f[i][1];
      g[3*i+2] = atom->f[i][2];
    }
    
    // L-BFGS two-loop recursion
    for (int i = 0; i < 3 * atom->nlocal; i++) q[i] = g[i];
    int bound = std::min(step, m);
    for (int i = bound - 1; i >= 0; i--) {
      int idx = (k - 1 - i + m) % m;
      alpha_vec[idx] = rho[idx] * (s[idx][0] * q[0] + s[idx][1] * q[1] + s[idx][2] * q[2]);
      for (int j = 0; j < 3 * atom->nlocal; j++) q[j] -= alpha_vec[idx] * y[idx][j];
    }
    
    // Scaling of the initial Hessian approximation
    double gamma = 1.0;
    if (step > 0) {
      double ys = 0.0;
      double yy = 0.0;
      for (int i = 0; i < 3 * atom->nlocal; i++) {
        ys += y[(k-1)%m][i] * s[(k-1)%m][i];
        yy += y[(k-1)%m][i] * y[(k-1)%m][i];
      }
      gamma = ys / yy;
    }
    
    for (int i = 0; i < 3 * atom->nlocal; i++) q[i] *= gamma;
    
    for (int i = 0; i < bound; i++) {
      int idx = (k - bound + i + m) % m;
      double beta = rho[idx] * (y[idx][0] * q[0] + y[idx][1] * q[1] + y[idx][2] * q[2]);
      for (int j = 0; j < 3 * atom->nlocal; j++) q[j] += s[idx][j] * (alpha_vec[idx] - beta);
    }
    
    // Update positions
    for (int i = 0; i < atom->nlocal; i++) {
      atom->x[i][0] -= alpha * q[3*i];
      atom->x[i][1] -= alpha * q[3*i+1];
      atom->x[i][2] -= alpha * q[3*i+2];
    }
    
    // Check for convergence
    double max_force = 0.0;
    for (int i = 0; i < atom->nlocal; i++) {
      double fx = atom->f[i][0];
      double fy = atom->f[i][1];
      double fz = atom->f[i][2];
      double force = sqrt(fx*fx + fy*fy + fz*fz);
      if (force > max_force) max_force = force;
    }
    if (max_force < tol) break;
    
    // Update s and y
    for (int i = 0; i < atom->nlocal; i++) {
      s[k%m][3*i] = atom->x[i][0] - x_old[3*i];
      s[k%m][3*i+1] = atom->x[i][1] - x_old[3*i+1];
      s[k%m][3*i+2] = atom->x[i][2] - x_old[3*i+2];
      y[k%m][3*i] = atom->f[i][0] - g[3*i];
      y[k%m][3*i+1] = atom->f[i][1] - g[3*i+1];
      y[k%m][3*i+2] = atom->f[i][2] - g[3*i+2];
    }
    
    double ys = 0.0;
    for (int i = 0; i < 3 * atom->nlocal; i++) ys += y[k%m][i] * s[k%m][i];
    rho[k%m] = 1.0 / ys;
    
    k++;
  }
  
  // Free allocated memory
  for (int i = 0; i < m; i++) {
    delete[] s[i];
    delete[] y[i];
  }
  delete[] s;
  delete[] y;
  delete[] rho;
  delete[] alpha_vec;
  delete[] g;
  delete[] g_old;
  delete[] x_old;
  delete[] q;
}