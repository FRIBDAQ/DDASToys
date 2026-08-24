/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Aaron Chester
             FRIB
             Michigan State University
             East Lansing, MI 48824-1321
*/

/**
 * @file  FitEditorCudaPSO.h
 * @brief Definition of the FitEditor class for GPU swarm (PSO/DE) fitting.
 */

#ifndef FITEDITORCUDAPSO_H
#define FITEDITORCUDAPSO_H

#include <CBuiltRingItemEditor.h>

#include <vector>

namespace ddasfmt {
class DDASHit;
}

/** @namespace ddastoys */
namespace ddastoys {

class Configuration;

/**
 * @class FitEditorCudaPSO
 * @brief Fit trace data on the GPU using the libCudaOptimize swarm optimizer
 * (see cudafit_analytic) and populate a hit extension with the results.
 * @details
 * Unlike FitEditorAnalytic, this editor always performs *both* the single- and
 * double-pulse fits and keeps the trace resident on the GPU between them
 * (cudafit1 leaves it loaded, cudafit2 reuses and frees it). Extending the hit
 * overwrites any existing extension. Intended for use with the EventEditor
 * framework. This is a development/experimental editor.
 * @note The underlying swarm optimizer uses file-scoped GPU state and is NOT
 * thread-safe; run single-threaded or under MPI (separate processes), not with
 * the ZMQ threaded strategy.
 */

class FitEditorCudaPSO : public CBuiltRingItemEditor::BodyEditor {
public:
  /** @brief Constructor. */
  FitEditorCudaPSO();
  /**
   * @brief Copy constructor.
   * @param rhs Object to copy construct.
   */
  FitEditorCudaPSO(const FitEditorCudaPSO &rhs);
  /**
   * @brief Move constructor.
   * @param rhs Object to move construct.
   */
  FitEditorCudaPSO(FitEditorCudaPSO &&rhs) noexcept;

  /**
   * @brief Copy assignment operator.
   * @param rhs Object to copy assign.
   * @return Reference to created object.
   */
  FitEditorCudaPSO &operator=(const FitEditorCudaPSO &rhs);
  /**
   * @brief Move assignment operator.
   * @param rhs Object to move assign.
   * @return Reference to created object.
   */
  FitEditorCudaPSO &operator=(FitEditorCudaPSO &&rhs) noexcept;

  /** @brief Destructor. */
  virtual ~FitEditorCudaPSO();

  // Mandatory interface from CBuiltRingItemEditor::BodyEditor
public:
  /**
   * @brief Perform the fit and create a fit extension for a single fragment.
   * @param pHdr     Pointer to the ring item header of the hit.
   * @param pBHdr    Pointer to the body header pointer for the hit.
   * @param bodySize Number of bytes in the body.
   * @param pBody    Pointer to the body.
   * @return         Final segment descriptors.
   */
  virtual std::vector<CBuiltRingItemEditor::BodySegment>
  operator()(pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize,
             void *pBody);
  /**
   * @brief Free the dynamic fit extension descriptor(s).
   * @param e IOvec we need to free.
   */
  virtual void free(iovec &e);

  // Private member data
private:
  Configuration *m_pConfig; //!< Configuration file parser.
};

} // namespace ddastoys

#endif
