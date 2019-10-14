

#ifndef JACOBIAN_H
#define JACOBIAN_H

/**
 * @file jacobian.h
 * @brief Provides class definitions for families of engines to support lmfit.
 */

/**
 * The concept is that each GSL lmfitter has to supply a pair of methods:
 * One that computes a vector of residuals and one that computes a Jacobian
 * matrix of partial derivatives.
 * At the implementation level we have two types of fits we need done:
 *    Single pulse fits and double pulse fits (the engines with names ending in 1 or 2).
 * For each fit type we have two fit engines:
 *    Serial computation (the engines with names starting with Serial)
 *    GPU accelerated (the engines with names starting with Cuda).
 *
 *  Finally a fit factory can generate the appropriate fit engine as desired
 *  by the actual fit.
 *
 */
#include <vector>
#include <stdint.h>


#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multifit_nlin.h>

// Abstract base class.

class FitEngine {
 protected:
  std::vector<uint16_t> x;           // Trace x coords
  std::vector<uint16_t> y;           // Trace y coords
 public:
  FitEngine(std::vector<std::pair<uint16_t, uint16_t>>&  data);
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J) = 0;
  virtual void residuals(const gsl_vector*p, gsl_vector* r)  = 0;
};

// concrete classes:

class SerialFitEngine1 : public FitEngine {
  SerialFitEngine1(std::vector<std::pair<uint16_t, uint16_t>>&  data);
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
};

class CudaFitEngine1 : public FitEngine {
  CudaFitEngine1(std::vector<std::pair<uint16_t, uint16_t>>&  data);
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
};

class SerialFitEngine2 : public FitEngine {
  SerialFitEngine2(std::vector<std::pair<uint16_t, uint16_t>>&  data);
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
};

class CudaFitEngine2 : public FitEngine {
  CudaFitEngine2(std::vector<std::pair<uint16_t, uint16_t>>&  data);
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
};

class FitEngineFactory {
  FitEngine* createSerialFitEngine1(std::vector<std::pair<uint16_t, uint16_t>>&  data);
  FitEngine*  createCudaFitEngine1(std::vector<std::pair<uint16_t, uint16_t>>&  data);

  FitEngine*  createSerialFitEngine2(std::vector<std::pair<uint16_t, uint16_t>>&  data);
  FitEngine*  createCudaFitEngine2(std::vector<std::pair<uint16_t, uint16_t>>&  data);

};


#endif
