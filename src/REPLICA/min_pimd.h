#ifdef MinClass
MinimizeStyle(pimd,MinPIMD)
#else
#ifndef LMP_STYLE_MIN_H
#define LMP_STYLE_MIN_H

#include "min.h"

namespace LAMMPS_NS {

class MinPIMD : public Min {
 public:
  MinPIMD(class LAMMPS *);
  void init();
  void run(int);
  void force_clear();
  void force_compute();

 private:
  class FixPIMDLangevin *fix_pimd;
};

}    // namespace LAMMPS_NS

#endif
#endif
