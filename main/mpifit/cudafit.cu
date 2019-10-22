/**
 * @author Ron Fox<fox@nscl.msu.edu>
 * @file cudafit.cu
 * @brief Provide trace fitting using the libucdafit library.
 * @note  We provide call compatible interfaces with lmfit1 and lmfit2, 
 * @note  This fit will not thread due to libcudaoptimize's need for us to 
 *        have global data for the device pointers to the trace.
 */

#include "lmfit.h"             // For the fit extension formats.
#include "reductions.cu"
#include <limits>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <DE_Optimizer.h>      // Optimizer beast.

// Define the parameter numbers for the fits:

static const unsigned A1 = 0;
static const unsigned K1 = 1;	// rise steepness
static const unsigned K2 = 2;	// exponential decay
static const unsigned X1 = 3;
static const unsigned C  = 4;

static const unsigned P1_NPARAMS = 5;

static const unsigned A2 = 5;
static const unsigned K3 = 6;
static const unsigned K4 = 7;
static const unsigned X2 = 8;

static const unsigned P2_NPARAMS = 9;

/**
 *  Here's why we can't have good things (threadable).  The libcudaoptimizer does not let me
 *  (to my knowledge) pass a parameter to my fitness function so I don't know how to get this
 *  information to it other than making it file scoped which is inherently thread-unsafe.
 */

static unsigned short* d_xCoords;        // trace x coordinates.
static unsigned short* d_yCoords;        // trace y coordinates.
static unsigned        n_tracePoints;  // Number of points in the trace.
static float*          h_pWeights(0);    // Host weights pointer.
static float*          d_pWeights(0);    // Device weights pointer.

/**
 * reportCudaError
 *   Report the most recent Cuda error as an std::runtime_error
 * @param context - describes the error context.
 */
static void
reportCudaError(const char* context)
{
  std::string msg("Error: ");
  msg += context;
  msg += " : ";
  cudaError_t status = cudaGetLastError();
  msg += cudaGetErrorString(status);
  throw std::runtime_error(msg);

}

/**
 * traceToGPU
 *   using the limits and saturation values to suppress some trace points
 *   Generates the x/y coordinates of the tracea that's left.
 * @param trace - raw trace.
 * @param limits - Left/right limits of thet race.
 * @param saturation - saturation values for the trace (values >= to this are eliminated).
 * @return - final number of points:
 */
static unsigned traceToGPU(
   std::vector<uint16_t> trace, std::pair<unsigned, unsigned> limits,
   uint16_t saturation
)
{
  std::vector<uint16_t> xcoords;
  std::vector<uint16_t> ycoords;

  int result(0);
  for (int i = limits.first; i < limits.second; i++) {
    if (trace[i] < saturation) {
      xcoords.push_back(i);
      ycoords.push_back(trace[i]);
      result++;
    }
  }
  // Allocate a pair of unsigned short device arrays:   d_xCoords and d_yCoords
  // and move the data from xcoords and ycoords into them:

  if (cudaMalloc(&d_xCoords, xcoords.size()*sizeof(unsigned short)) != cudaSuccess) {
    reportCudaError("Allocating GPU memory for trace x-coordinates");
  }
  if (cudaMalloc(&d_yCoords, ycoords.size()*sizeof(unsigned short)) != cudaSuccess) {
    reportCudaError("Allocating GPU memory for trace y-coordinates");
  }

  if (cudaMemcpy(
      d_xCoords, xcoords.data(), xcoords.size()*sizeof(unsigned short), cudaMemcpyHostToDevice)
      != cudaSuccess) {
    reportCudaError("Moving trace x coordinates into the GPU");
  }
  if (cudaMemcpy(
      d_yCoords, ycoords.data(), ycoords.size()*sizeof(unsigned short), cudaMemcpyHostToDevice)
      != cudaSuccess) {
    reportCudaError("Moving trace y coordinates into the GPU");
  }
  // We'll use weights of 1.0;   This can be modified here:

  h_pWeights = static_cast<float*>(malloc(result * sizeof(float)));
  for (int i =0; i < result; i++) {
    h_pWeights[i] = 1.0;
  }
  if(!cudaMalloc(&d_pWeights, result*sizeof(float) != cudaSuccess)) {
    reportCudaError("Failed to allocates device weights array");
  }
  if (cudaMemcpy(d_pWeights, h_pWeights, result*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    reportCudaError("Failed to copy wieghts into the device");
  }

  n_tracePoints = result;
  return result;
}
/**
 *  freeTrace
 *     Release the GPU memory associated with the trace:
 */
static void
freeTrace()
{
  cudaFree(d_xCoords);
  cudaFree(d_yCoords);
  cudaFree(d_pWeights);
  free(h_pWeights);
}
/**
 * logistic - GPU FUNCTION!!!
 *    Evaluate a logistic function for the specified parameters and point.
 *    A logistic function is a function with a sigmoidal shape.  We use it
 *    to fit the rising edge of signals DDAS digitizes from detectors.
 *    See e.g. https://en.wikipedia.org/wiki/Logistic_function for
 *    a discussion of this function.
 *
 * @param A  - Amplitude of the signal.
 * @param k1 - steepness of the signal (related to the rise time).
 * @param x1 - Mid point of the rise of the sigmoid.
 * @param x  = Location at which to evaluate the function.
 * @return double
 */
__device__ float
logistic(float A, float  k, float x1, float x)
{
    return A/(1+expf(-k*(x-x1)));
}

/**
 * decay  - GPU FUNCTION!!!
 *    Signals from detectors usually have a falling shape that approximates
 *    an exponential.  This function evaluates this decay at some point.
 *
 *  @param A1 - amplitude of the signal
 *  @param k1 - Decay time factor f the signal.
 *  @param x1 - Position of the pulse.
 *  @param x  - Where to evaluate the signal.
 *  @return double
 */
__device__ float
decay(float A, float k, float  x1, float x)
{
    return A*(expf(-k*(x-x1)));
}


/**
 * singlePulse -- GPU Function
 *    Evaluate the value of a single pulse in accordance with our
 *    canonical functional form.  The form is a sigmoid rise with an
 *    exponential decay that sits on top of a constant offset.
 *    The exponential decay is turned on with switchOn() above when
 *    x > the rise point of the sigmoid.
 *
 * @param A1  - pulse amplitiude
 * @parm  k1  - sigmoid rise steepness.
 * @param k2  - exponential decay time constant.
 * @param x1  - sigmoid position.
 * @param C   - Constant offset.
 * @param x   - Position at which to evaluat this function
 * @return double
 */
__device__ float
singlePulse(
    float A1, float  k1, float  k2, float x1, float  C, float  x
)
{
    return (logistic(A1, k1, x1, x)  * decay(1.0, k2, x1, x)) // decay term
        + C;                                        // constant.
}
/**
 * doublePulse - GPU FUNCTION!!!
 *    Evaluate the canonical form of a double pulse.  This is done
 *    by summing two single pulses.  The constant term is thrown into the
 *    first pulse.  The second pulse gets a constant term of 0.
 *
 * @param A1   - Amplitude of the first pulse.
 * @param k1   - Steepness of first pulse rise.
 * @param k2   - Decay time of the first pulse.
 * @param x1   - position of the first pulse.
 *
 * @param A2   - Amplitude of the second pulse.
 * @param k3   - Steepness of second pulse rise.
 * @param k4   - Decay time of second pulse.
 * @param x2   - position of second pulse.
 *
 * @param C    - Constant offset the pulses sit on.
 * @param x    - position at which to evaluate the pulse.
 * @return double.
 * 
*/
__device__ float
doublePulse(
    float A1, float k1, float k2, float x1,
    float A2, float k3, float  k4, float  x2,
    float  C, float  x    
)
{
    float  p1 = singlePulse(A1, k1, k2, x1, C, x);
    float  p2 = singlePulse(A2, k3, k4, x2, 0.0, x);
    return p1 + p2;
}

/**
 * chiFitness1  -- GPU FUNCTION!!!!
 *
 *   Computes the chisquare fitness for one point in one solution given that d_fitness
 *   has pulled out what we need.  This fitness is for a single pulse fit.
 *   @param pParams - pointer to this solutions parameters.
 *   @param x       - X coordinate.
 *   @param y       - Y coordinate.
 *   @param wt      - weight for this coordinate (for now unused).
 *   @return float  - square of difference between solution and actual.
 */
__device__
float chiFitness1(const float* pParams, float x, float y, float wt)
{
  // Get the parameters from the fit:

  float a  = pParams[A1];
  float k1 = pParams[K1];
  float k2 = pParams[K2];
  float x1 = pParams[X1];
  float c = pParams[C];

  float fit = singlePulse(a, k1, k2, x1, c, x);
  float d   = (y  - fit);
  return d*d;

  
}
/**
 * chiFitness2 -- GPU FUNCTION
 *   Cmoputes the chi squre fitness contribution for one point in one solution
 *   given that our caller has pulled out what we need:
 *   @param pParams - pointer to this solutions parameters.
 *   @param x       - X coordinate.
 *   @param y       - Y coordinate.
 *   @param wt      - weight for this coordinate (for now unused).
 *   @return float  - square of difference between solution and actual.
 */
__device__
float chiFitness2(const float* pParams, float x, float y, float wt)
{
  // Get the parameters from the fit:

  float a1 = pParams[A1];
  float a2 = pParams[A2];
  float k1 = pParams[K1];
  float k3 = pParams[K3];
  float k2 = pParams[K2];
  float k4 = pParams[K4];
  float x1 = pParams[X1];
  float x2 = pParams[X2];
  float c  = pParams[C];

  float fit = doublePulse(a1, k1, k2, x1, a2, k3, k4, x2, c, x);
  float d   = y - fit;
  return d*d;

}

/**
 * d_fitness1   -- GPU FUNCTION!!!!
 *   This function lives in the GPU and:
 *   - Computes the chi-square contribution for a single point for a single solution 
 *     in the swarm for a single pulse with an offset.
 *   - Uses reduceToSum to sum the chisquare contributions over the entire
 *     trace.
 *   The result is put into the fitness value for our solution.
 *
 *  @param pSolutions - pointer to solutions array in the GPU.
 *  @param pFitnesses - pointer to the array of fitnesse for all solutions in the swarm.
 *  @param nParams    - Number of parameters in the fit (should be 5).
 *  @param nSol       - Number of solutions in the swarm.
 *  @param pXcoords   - Trace xcoordinates array.
 *  @param pYcoords   - Trae y coordinates array.
 *  @param pWeights   - Y weights to apply.
 *  @param nPoints    - Number of points in the trace.
 *
 */
__global__
void d_fitness1(const float* pSolutions, float* pFitnesses, int nParams, int nSol,
	       unsigned short* pXcoords, unsigned short* pYcoords, float* pWeights,
	       int nPoints)
{
  extern __shared__ float sqdiff[];  // Locate the chisqr contribs in shared mem.


  // Figure out which solution and point we're working on.  This is based 
  // on our place in the computation's geometry:

  int swarm = blockIdx.x;
  int solno = blockIdx.y + swarm*nSol; // Our solution.
  int ptno  = threadIdx.x;	      // Our point.

  if ((solno <  nSol*gridDim.x) && (ptno < nPoints)) {
    int ipt = ptno + swarm*nPoints;
    float x = pXcoords[ipt];
    float y = pYcoords[ipt];
    sqdiff[ptno]  = chiFitness1(pSolutions + (solno*nParams), x, y, 1.0);

    // Can't do the fanin sum until all threads have computed:

    __syncthreads();

    reduceToSum<float, MAXPOINTS>(sqdiff, ptno);
    __syncthreads();   // The sum is now done into sqdiff[0]:
    if(ptno == 0) {
      pFitnesses[solno] = sqdiff[0];
    }
  }
  
}
/**
 *  d_fitness2  - GPU FUNCTION!!!
 *    Compute the chisquare fitness for one point of one solution in the swarm.
 *    Once that's done in all threads, we fire off our part of a fan-in parallel
 *    sum over our solution.
 *    Much of what we do is figure out our place in the world so that we can pass
 *    the right stuff to chiFitness2 which does the actual computation.
 *
 *  @param pSolutions - pointer to solutions array in the GPU.
 *  @param pFitnesses - pointer to the array of fitnesse for all solutions in the swarm.
 *  @param nParams    - Number of parameters in the fit (should be 5).
 *  @param nSol       - Number of solutions in the swarm.
 *  @param pXcoords   - Trace xcoordinates array.
 *  @param pYcoords   - Trae y coordinates array.
 *  @param pWeights   - Y weights to apply.
 *  @param nPoints    - Number of points in the trace.
 *
 */
__global__
void d_fitness2(const float* pSolutions, float* pFitnesses, int nParams, int nSol,
	       unsigned short* pXcoords, unsigned short* pYcoords, float* pWeights,
	       int nPoints)
{
  extern __shared__ float sqdiff[];  // Locate the chisqr contribs in shared mem.


  // Figure out which solution and point we're working on.  This is based 
  // on our place in the computation's geometry:

  int swarm = blockIdx.x;
  int solno = blockIdx.y + swarm*nSol; // Our solution.
  int ptno  = threadIdx.x;	      // Our point.

  if ((solno <  nSol*gridDim.x) && (ptno < nPoints)) {
    int ipt = ptno + swarm*nPoints;
    float x = pXcoords[ipt];
    float y = pYcoords[ipt];
    sqdiff[ptno]  = chiFitness2(pSolutions + (solno*nParams), x, y, 1.0);

    // Can't do the fanin sum until all threads have computed:

    __syncthreads();

    reduceToSum<float, MAXPOINTS>(sqdiff, ptno);
    __syncthreads();   // The sum is now done into sqdiff[0]:
    if(ptno == 0) {
      pFitnesses[solno] = sqdiff[0];
    }
  }
  
}


/**
 * h_fitSingle
 *    Invokes the kernel that produces the fitness measure.
 *    The fitness is computed in the GPU and is the chi square.
 *
 * @param solutions - pointer to the Cuda solution set.
 * @param fitnesses - pointer to the current fitness set
 * @param grid      - Computational grid being used.
 * @param block     - Shapes of blocks within the grid.
 */
void
h_fitSingle(
   const CudaOptimize::SolutionSet* solutions, CudaOptimize::FitnessSet* fitnesses,
   dim3 grid, dim3 block
)
{
  const float*   d_solutions = solutions->getDevicePositionsConst();    // Current solutions.
  float*         d_fitnesses = fitnesses->get();                        // Where fitnesses go.

  // Figure out how many warps the fitnesses require:

  int nParams = solutions->getProblemDimension();
  nParams     = (nParams + 31)/32;
  nParams     = nParams*32;

  // Which solution:

  int nsol = solutions->getSolutionNumber();

  // Figure out the bocksize of the computation:

  dim3 myBlockSize(n_tracePoints, 1, 1);
  d_fitness1<<< grid, myBlockSize, n_tracePoints*sizeof(float) >>>(
    d_solutions, d_fitnesses, nParams, nsol, d_xCoords, d_yCoords, d_pWeights, n_tracePoints
  );
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    reportCudaError("Failed to run single pulsse fitness kernel");
  }

}


/**
 * h_fitDouble
 *   Host part to setup computation of the fitnesses across the swarm for our
 *   fits for a double pulse.  Really this just sets up the
 *   kernel call for fitness2 which does the rest.
 * 
 * @param solutions - pointer to the current Solution set.
 * @param fitnessses - pointer to the current fitness set
 * @param grid      - Computaional grid geometry.
 * @param block     - Shapes of the blocks within the grid.
 */
void
h_fitDouble(
   const CudaOptimize::SolutionSet* solutions, CudaOptimize::FitnessSet* fitnesses,
   dim3 grid, dim3 block
)
{
  const float*   d_solutions = solutions->getDevicePositionsConst();    // Current solutions.
  float*         d_fitnesses = fitnesses->get();                        // Where fitnesses go.

  // Figure out how many warps the fitnesses require:

  int nParams = solutions->getProblemDimension();
  nParams     = (nParams + 31)/32;
  nParams     = nParams*32;

  // Which solution:

  int nsol = solutions->getSolutionNumber();

  // Figure out the bocksize of the computation:

  dim3 myBlockSize(n_tracePoints, 1, 1);
  d_fitness2<<< grid, myBlockSize, n_tracePoints*sizeof(float) >>>(
    d_solutions, d_fitnesses, nParams, nsol, d_xCoords, d_yCoords, d_pWeights, n_tracePoints
  );
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    reportCudaError("Failed to run single pulsse fitness kernel");
  }
  
}

/**
 * cudafit1
 *   Fit a single pulse to the data:
 * @param pResult - pointer to the resulting parameters.
 * @param trace   - references the raw trace data.
 * @param limits  - Provides the limits over which the trace is done.
 * @param saturation - Defines the FADC saturation level.
 * @param freeTraceWhenDone - if true (default) the trace data is freed from the GPU
 *                     if not it's left allocated.  This allows a double fit to be done
 *                     immediately after with no reallocation/copy.
 */
void
cudafit1(
	 DDAS::fit1Info* pResult, const std::vector<uint16_t>& trace,
	 const std::pair<unsigned, unsigned>& limits,
	 uint16_t saturation = 0xffff, bool freeTraceWhenDone=true
)
{
  size_t nPoints = traceToGPU(trace, limits, saturation);

  // Create and setup the optimizer - fitness function will be done in the device:

  CudaOptimize::DE_Optimizer opt(&h_fitSingle, P1_NPARAMS, 1, 200);   // last parameter the swarmsize?
  opt.setTerminationFlags((CudaOptimize::TERMINATION_FLAGS)(CudaOptimize::TERMINATE_GENS | CudaOptimize::TERMINATE_FIT));
  opt.setGenerations(100); 
  opt.setStoppingFitness(10.0);
  opt.setMutation(CudaOptimize::DE_RANDOM);
  opt.setCrossover(CudaOptimize::DE_BINOMIAL);
  opt.setHostFitnessEvaluation(false);


  // Set constraints on the parameters.

  opt.setBounds(0, A1, make_float2(saturation*10, 0.0));
  opt.setBounds(0, K1, make_float2(500.0, 0.0));
  opt.setBounds(0, K2, make_float2(500.0, 0.0));
  opt.setBounds(0, X1, make_float2(-50.0, nPoints+50));    // Let the positions go a bit before/past the trace.
  opt.setBounds(0, C,  make_float2(saturation/4.0, 0.0));  // 25% full scale offset should be generous.
  
  opt.optimize();

  if (freeTraceWhenDone) freeTrace();

  // Pull out the fit values into the pResult.

  pResult->chiSquare =  opt.getBestFitness(0);
  pResult->fitStatus =  0;
  pResult->iterations = opt.getFunctionEvals();	// closest to an iteration count we have.
  float* pParams      = opt.getBestSolution(0);
  pResult->offset     = pParams[C];
  pResult->pulse.position = pParams[X1];
  pResult->pulse.amplitude= pParams[A1];
  pResult->pulse.steepness= pParams[K1];
  pResult->pulse.decayTime = pParams[K2];


}
/**
 * cudafit2
 *   Two a double pulse fit using libcudaoptimize.
 *
 * @param pResult - pointer to the resulting parameters.
 * @param trace   - references the raw trace data.
 * @param limits  - Provides the limits over which the trace is done.
 * @param saturation - Defines the FADC saturation level.
 * @param traceIsLoaded - if true, the trace is already loaded into the GPU
 *                   from a prior cudafit1 call.  Note that regardless the trace is freed
 *                   after we're run.  The default requires us to copy the trace.
 */
void
cudafit2(
	 DDAS::fit2Info* pResult, const std::vector<uint16_t>& trace,
	 const std::pair<unsigned, unsigned>& limits,
	 uint16_t saturation = 0xffff, bool traceIsLoaded = false
)
{
  // If needed get the trace into the GPU

  size_t nPoints;
  if (traceIsLoaded) {
    nPoints = n_tracePoints;                     // From prior load.
  } else {
    nPoints = traceToGPU(trace, limits, saturation);
  }

  // Set up the optimizer with the fitness done in the GPU:

  CudaOptimize::DE_Optimizer opt(h_fitDouble, P2_NPARAMS, 1, 200);
  opt.setTerminationFlags((CudaOptimize::TERMINATION_FLAGS)(CudaOptimize::TERMINATE_GENS | CudaOptimize::TERMINATE_FIT));
  opt.setGenerations(100); 
  opt.setStoppingFitness(10.0);
  opt.setMutation(CudaOptimize::DE_RANDOM);
  opt.setCrossover(CudaOptimize::DE_BINOMIAL);
  opt.setHostFitnessEvaluation(false);

  // Constrain the parameters - unfortunately we can't constrain x1 < x2 :-(
  // We give corresponding parameters in the second pulse the same constraints.

  opt.setBounds(0, A1, make_float2(saturation*10, 0.0));
  opt.setBounds(0, A2, make_float2(saturation*10, 0.0));
  opt.setBounds(0, K1, make_float2(500.0, 0.0));
  opt.setBounds(0, K3, make_float2(500.0, 0.0));
  opt.setBounds(0, K2, make_float2(500.0, 0.0));
  opt.setBounds(0, K4, make_float2(500.0, 0.0));
  opt.setBounds(0, X1, make_float2(-50.0, nPoints+50));    // Let the positions go a bit before/past the trace.
  opt.setBounds(0, X2, make_float2(-50.0, nPoints+50));    // Let the positions go a bit before/past the trace.
  opt.setBounds(0, C,  make_float2(saturation/4.0, 0.0));  // 25% full scale offset should be generous.

  opt.optimize();

  freeTrace();                                             // Always!!
  
  // We only allowed one case so pull the best fitness and best solution from it:

  pResult->chiSquare = opt.getBestFitness(0);
  pResult->fitStatus = 0;
  pResult->iterations= opt.getFunctionEvals();
  float * pParams    = opt.getBestSolution(0);
  pResult->offset    = pParams[C];

  pResult->pulses[0].position = pParams[X1];
  pResult->pulses[0].amplitude= pParams[A1];
  pResult->pulses[0].steepness= pParams[K1];
  pResult->pulses[0].decayTime = pParams[K2];

  pResult->pulses[1].position = pParams[X2];
  pResult->pulses[1].amplitude= pParams[A2];
  pResult->pulses[1].steepness= pParams[K3];
  pResult->pulses[1].decayTime = pParams[K4];


 
}
