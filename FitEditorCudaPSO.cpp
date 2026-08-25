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
 * @file  FitEditorCudaPSO.cpp
 * @brief Implementation of the FitEditor class for GPU swarm (PSO/DE) fitting.
 */

#include "FitEditorCudaPSO.h"

#include <iostream>
#include <stdexcept>
#include <utility>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "Configuration.h"
#include "cudafit_analytic.cuh"
#include "fit_extensions.h"
#include "profiling.h"

using namespace ddasfmt;
using namespace ddastoys;

#ifdef ENABLE_TIMING
static Stats stats;
static double total(0);
#endif

/**
 * @details
 * Sets up the configuration manager to parse config files and manage
 * configuration data. Reads the fit config file.
 */
ddastoys::FitEditorCudaPSO::FitEditorCudaPSO() : m_pConfig(new Configuration) {
  try {
    m_pConfig->readConfigFile();
  } catch (std::exception &e) {
    std::cerr << "Error configuring FitEditorCudaPSO: " << e.what()
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

ddastoys::FitEditorCudaPSO::FitEditorCudaPSO(const FitEditorCudaPSO &rhs)
    : m_pConfig(new Configuration(*rhs.m_pConfig)) {}

/**
 * @details
 * Constructs using move assignment.
 */
ddastoys::FitEditorCudaPSO::FitEditorCudaPSO(FitEditorCudaPSO &&rhs) noexcept
    : m_pConfig(nullptr) {
  *this = std::move(rhs);
}

FitEditorCudaPSO &
ddastoys::FitEditorCudaPSO::operator=(const FitEditorCudaPSO &rhs) {
  if (this != &rhs) {
    delete m_pConfig;
    m_pConfig = new Configuration(*rhs.m_pConfig);
  }

  return *this;
}

FitEditorCudaPSO &
ddastoys::FitEditorCudaPSO::operator=(FitEditorCudaPSO &&rhs) noexcept {
  if (this != &rhs) {
    delete m_pConfig;
    m_pConfig = rhs.m_pConfig;
    rhs.m_pConfig = nullptr;
  }

  return *this;
}

/**
 * @details
 * Delete the Configuration object managed by this class.
 */
ddastoys::FitEditorCudaPSO::~FitEditorCudaPSO() { delete m_pConfig; }

/**
 * @details
 * This is the hook into the FitEditorCudaPSO class. Here we:
 * - Parse the fragment into a hit.
 * - Produce an IOvec element for the existing hit (without any fit that might
 *   have been there).
 * - See if the configuration manager says we should fit and if so, create the
 *   trace.
 * - Verify the trace length matches the configuration.
 * - Always perform both the single- and double-pulse swarm fits, keeping the
 *   trace resident on the GPU between them.
 * - Create an IOvec entry for the extension we created (dynamic).
 */
std::vector<CBuiltRingItemEditor::BodySegment>
ddastoys::FitEditorCudaPSO::operator()(pRingItemHeader pHdr, pBodyHeader pBHdr,
                                       size_t bodySize, void *pBody) {
  std::vector<CBuiltRingItemEditor::BodySegment> result;

  // Regardless we want a segment that includes the hit. Note that the first
  // uint32_t of the body is the size of the standard hit part in uint16_t
  // words.

  uint32_t *pSize = static_cast<uint32_t *>(pBody);
  CBuiltRingItemEditor::BodySegment hitInfo(*pSize * sizeof(uint16_t), pSize,
                                            false);
  result.push_back(hitInfo);

  // Make the hit:

  DDASHit hit;
  DDASHitUnpacker unpacker;
  unpacker.unpack(static_cast<uint32_t *>(pBody),
                  static_cast<uint32_t *>(nullptr), hit);

  auto crate = hit.getCrateID();
  auto slot = hit.getSlotID();
  auto chan = hit.getChannelID();

  if (m_pConfig->fitChannel(crate, slot, chan)) {
    std::vector<uint16_t> trace = hit.getTrace();
    FitInfo *pFit = new FitInfo; // Have an extension tho may be zero.

    if (trace.size() > 0) { // Need a trace to fit
      // Verify that the trace length is what the configuration file expects:
      auto expectedLength = m_pConfig->getTraceLength(crate, slot, chan);
      if (trace.size() != expectedLength) {
        std::cerr << "Trace length mismatch for crate " << crate << " slot "
                  << slot << " channel " << chan << " expected "
                  << expectedLength << " got " << trace.size() << std::endl;
        throw std::length_error("Trace length mismatch");
      }

      auto limits = m_pConfig->getFitLimits(crate, slot, chan);
      auto sat = m_pConfig->getSaturationValue(crate, slot, chan);

      // Always do both fits. Keep the trace resident on the GPU between them:
      // cudafit1 leaves it loaded (freeTraceWhenDone = false), cudafit2 reuses
      // it (traceIsLoaded = true) and frees it when done.

#ifdef ENABLE_TIMING
      Timer timer;
#endif
      analyticfit::cudafit1(&(pFit->s_extension.onePulseFit), trace, limits,
                            sat, /*freeTraceWhenDone=*/false);
      analyticfit::cudafit2(&(pFit->s_extension.twoPulseFit), trace, limits,
                            sat, /*traceIsLoaded=*/true);
#ifdef ENABLE_TIMING
      stats.addData(timer.elapsed());
      if (stats.size() == 1000) {
        stats.compute();
        stats.print("======== CUDA PSO stats ========");
      }
#endif
    }

    CBuiltRingItemEditor::BodySegment fit(sizeof(FitInfo), pFit, true);
    result.push_back(fit);

  } else { // No fit performed
    nullExtension *p = new nullExtension;
    CBuiltRingItemEditor::BodySegment nofit(sizeof(nullExtension), p, true);
    result.push_back(nofit);
  }

  return result;
}

void ddastoys::FitEditorCudaPSO::free(iovec &e) {
  if (e.iov_len == sizeof(FitInfo)) {
    FitInfo *pFit = static_cast<FitInfo *>(e.iov_base);
    delete pFit;
  } else {
    nullExtension *p = static_cast<nullExtension *>(e.iov_base);
    delete p;
  }
}

/////////////////////////////////////////////////////////////////////////////
// Factory for our editor:
//

/**
 * @brief Factory method to create this FitEditor.
 *
 * @details
 * $DAQBIN/EventEditor expects a symbol called createEditor to exist in the
 * plugin library it loads at runtime. Wrapping the factory method in
 * extern "C" prevents namespace mangling by the C++ compiler.
 */
extern "C" {
ddastoys::FitEditorCudaPSO *createEditor() {
  return new ddastoys::FitEditorCudaPSO;
}
}
