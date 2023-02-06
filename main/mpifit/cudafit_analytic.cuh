/**
 *  @file cudafit.cuh 
 *  @brief header for the cuda swarm fitting code.
 */
#ifndef CUDAFIT_ANALYTIC_CUH
#define CUDAFIT_ANALYTIC_CUH
// Forward definitions.

namespace DDAS {
  struct fit1Info;
  struct fit2Info;
}

// fit one pulse:

extern void
cudafit1(
	 DDAS::fit1Info* pResult, const std::vector<uint16_t>& trace,
	 const std::pair<unsigned, unsigned>& limits,
	 uint16_t saturation = 0xffff, bool freeTraceWhenDone=true
);

// Fit 2 pulses.

extern void
cudafit2(
	 DDAS::fit2Info* pResult, const std::vector<uint16_t>& trace,
	 const std::pair<unsigned, unsigned>& limits,
	 uint16_t saturation = 0xffff, bool traceIsLoaded = false
);
#endif
