/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Aaron Chester
	     Bashir Sadeghi
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/**
 * @file mlinference.cpp
 * @brief Function implementations for machine-learning inference editor.
 */

#include "mlinference.h"

#include <algorithm>

#include <torch/torch.h>

#include "fit_extensions.h"

using namespace ddastoys;
using TT = torch::Tensor; //!< Just a QOL shorthand for tensors

/**
 * @todo (ASC 9/19/24): Does using input/output parameters for pre- and 
 * post-processing data improve performance (avoid multiple 
 * construciton/assignment)?
 */

/**
 * @todo (ASC 2/21/25): Number of samples for baseline estimation as a 
 * configurable parameter.
 */

/*------------------------------------------------------------------
 * Utility functions.
 */

/**
 * @brief Pre-process the input and return a torch tensor.
 * 
 * @param trace    References the trace to process.
 * @param nsamples Number of samples for baseline estimation to remove offset. 
 *
 * @return Pre-processed data as a PyTorch tensor.
 *
 * @details
 * Pre-processing involves two steps:
 * - Remove the DC offset.
 * - Normalizing the trace by dividing each sample by the max value following 
 *   baseline removal.
 * Note that the trace-as-a-normalized-tensor is floating point.
 *
 * @note (ASC 9/23/24): Adopted from 
 * https://github.com/bashir-sadeghi/frib/tree/main/pytorch_to_cpp
 * 
 * @note (ASC 2/21/25): The baseline estimation has changed with respect to 
 * previous versions of this code. Trace baseline is the average of the 
 * nsamples smallest ADC values on the trace. More robust in most cases 
 * (B. Sadeghi). This may not work so well in the presence of high-frequency 
 * noise or large baseline undershoots.
 */
static std::tuple<TT, TT, TT>
preprocessInput(std::vector<uint16_t>& trace, unsigned nsamples)
{    
    // Create torch tensor. We don't have kUint16 in this version of PyTorch,
    // so we use std::transform to read float data into the tensor directly:
    
    TT input = torch::empty({static_cast<long>(trace.size())},
			    torch::kFloat32);
    float* pData = input.data_ptr<float>(); // Points to first element
    std::transform(trace.begin(), trace.end(), pData,
		   [](const uint16_t& data) {
		       return static_cast<float>(data);
		   });

    // Remove the offset:
    
    TT slice = std::get<0>(torch::topk(input, nsamples, 0, false, false));
    TT offset = torch::mean(slice);
    input -= offset;
    
    // Divide the trace by its maximum value:

    TT max = torch::max(input);
    input /= max;
    
    return std::make_tuple(input, max, offset);
}

/**
 * @brief Post-process the input and return the results.
 *
 * @param[in,out] pResult Pointer to results storage.
 * @param[in] output References our processed tensor.
 * @param[in] xscale x-axis normalziation factor (the trace size).
 * @param[in] yscale y-axis normalization factor (max value).
 * @param[in] offset y-axis offset of amplitude-adjusted normalized trace.
 */
static void
postProcessOutput(
    FitInfo* pResult, c10::IValue& output, double xscale,
    double yscale, double offset
    )
{
    // Classification probability. First element is single pulse prob,
    // second element is double pulse prob:

    auto tup = output.toTuple();
    auto elements = tup->elements();
    
    TT logit = elements[1].toTensor();
    // TT pclass = torch::argmax(logit, 1); // 0: single pulse, 1: double pulse
    TT prob = torch::softmax(logit, 1);
    
    // Single pulse results:

    double t_s  = xscale * elements[2].toTensor().item<double>();
    double a_s  = yscale * elements[3].toTensor().item<double>();
    double k1_s = elements[5].toTensor().item<double>() / xscale;
    double k2_s = elements[6].toTensor().item<double>() / xscale;
	
    // Double pulse results:

    double t1_d = xscale * elements[2].toTensor().item<double>();
    double a1_d = yscale * elements[4].toTensor().item<double>();
    double k1_d = elements[5].toTensor().item<double>() / xscale;
    double k2_d = elements[6].toTensor().item<double>() / xscale;
    double t2_d = xscale * elements[7].toTensor().item<double>();
    double a2_d = yscale * elements[8].toTensor().item<double>();
    double k3_d = elements[9].toTensor().item<double>() / xscale;
    double k4_d = elements[10].toTensor().item<double>() / xscale;
    
    // Populate our result:
    
    pResult->s_extension.onePulseFit.pulse.amplitude = a_s;
    pResult->s_extension.onePulseFit.pulse.position  = t_s;
    pResult->s_extension.onePulseFit.pulse.decayTime = k1_s;
    pResult->s_extension.onePulseFit.pulse.steepness = k2_s;
    pResult->s_extension.onePulseFit.offset = offset;
    
    pResult->s_extension.twoPulseFit.pulses[0].amplitude = a1_d;
    pResult->s_extension.twoPulseFit.pulses[0].position  = t1_d;
    pResult->s_extension.twoPulseFit.pulses[0].decayTime = k1_d;
    pResult->s_extension.twoPulseFit.pulses[0].steepness = k2_d;
    pResult->s_extension.twoPulseFit.pulses[1].amplitude = a2_d;
    pResult->s_extension.twoPulseFit.pulses[1].position  = t2_d;
    pResult->s_extension.twoPulseFit.pulses[1].decayTime = k3_d;
    pResult->s_extension.twoPulseFit.pulses[1].steepness = k4_d;
    pResult->s_extension.twoPulseFit.offset = offset;
    
    pResult->s_extension.singleProb = prob[0][0].item<double>();
    pResult->s_extension.doubleProb = prob[0][1].item<double>();
}

/**
 * @details
 * This is the interface to perform the machine-learning inference fit on a 
 * single trace analogous to the lmfit functions for iterative fitting. 
 * The steps are pretty simple:
 * - Preprocess the trace (offest removal and normalization).
 * - Perform inference on normalized input.
 * - Postprocess the trace (rescaling, extract and store fit parameters).
 * Model is set to inference mode to disable gradient trackig and reduce
 * overhead.
 */
void
ddastoys::mlinference::performInference(
    FitInfo* pResult, std::vector<uint16_t>& trace, unsigned saturation,
    torch::jit::script::Module& module
    )
{
    // Inference mode: don't calculate gradients, histories, etc. which
    // are only needed for training:

    torch::InferenceMode mode;
    
    // Preprocess, store normalization constants for later:

    unsigned baselineSamples = 15; // How many samples for baseline removal?
    auto preprocessed = preprocessInput(trace, baselineSamples);

    // preprocessed elements are torch::Tensors. We extract the values for
    // the offset and max and cache them as doubles:
    
    auto normalizedInput = std::get<0>(preprocessed);
    auto max = std::get<1>(preprocessed).item<double>();
    auto offset = std::get<2>(preprocessed).item<double>();
    auto samples = static_cast<double>(trace.size());

    normalizedInput = normalizedInput.unsqueeze_(0); // trace.size() x 1 matrix
    
    // Inference step, input must be matrix:

    c10::IValue output = module.forward({normalizedInput});

    // Fish out and populate the results:

    postProcessOutput(pResult, output, samples, max, offset);
}
