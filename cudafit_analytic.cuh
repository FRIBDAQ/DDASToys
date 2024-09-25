/**
 *  @file  cudafit_analytic.cuh 
 *  @brief Header for the CUDA swarm fitting code.
 */

#ifndef CUDAFIT_ANALYTIC_CUH
#define CUDAFIT_ANALYTIC_CUH

#include <cstdint>
#include <vector>

/** @namespace ddastoys */
namespace ddastoys {
    struct fit1Info;
    struct fit2Info;

    /** @namespace ddastoys::analyticfit */
    namespace analyticfit {

	// Fit a single pulse:

	/**
	 * @brief Perform a single-pulse fit using libcudaoptimize.
	 * @param pResult    Pointer to the resulting parameters.
	 * @param trace      References the raw trace data.
	 * @param limits     Provides the limits over which the trace is done.
	 * @param saturation Defines the FADC saturation level.
	 * @param freeTraceWhenDone If true (default) the trace data is freed 
	 *   from the GPU if not it's left allocated. This allows a double fit 
	 *   to be done immediately after with no reallocation/copy.
	 */
	extern void
	cudafit1(
	    fit1Info* pResult, const std::vector<uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    uint16_t saturation=0xffff, bool freeTraceWhenDone=true
	    );

	// Fit two pulses:

	/**
	 * @brief Perform a double-pulse fit using libcudaoptimize.
	 *
	 * @param pResult    Pointer to the resulting parameters.
	 * @param trace      References the raw trace data.
	 * @param limits     Provides the limits over which the trace is done.
	 * @param saturation Defines the FADC saturation level.
	 * @param traceIsLoaded If true, the trace is already loaded into the 
	 *   GPU from a prior cudafit1 call. Note that regardless the trace is 
	 *   freed after we're run. The default requires us to copy the trace.
	 */
	extern void
	cudafit2(
	    ddastoys::fit2Info* pResult, const std::vector<uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    uint16_t saturation=0xffff, bool traceIsLoaded=false
	    );

    } // namespace cudafit
} // namespace ddastoys

#endif
