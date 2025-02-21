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
using TT = torch::Tensor; // Just QOL

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
 * @brief Given a trace and a saturation value, returns the vector of 
 * sample number, sample value pairs that are below saturation.
 *
 * @param trace      Raw trace.
 * @param saturation Saturation level.
 * 
 * @return Reduced trace: vector of sample number, sample value pairs.
 *
 * @note It is currently unclear whether this is needed/how it should be used. 
 * The infernce model is trained using all samples of unsaturated traces. 
 * Reducing the trace changes the input shape. What is the model performance 
 * on saturated traces?
 */
static std::vector<std::pair<uint16_t, uint16_t>>
reduceTrace(std::vector<uint16_t>& trace, unsigned saturation)
{
    std::vector<std::pair<uint16_t, uint16_t>> reduced;
    for (unsigned i = 0; i < trace.size(); i++) {
	if (trace[i] < saturation) {
	    reduced.push_back(std::pair<uint16_t, uint16_t>(i, trace[i]));
	}
    }
    
    return reduced;
}

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
    // Convert the trace into a torch tensor. Note that first we transform
    // the input to float and hope this is 32 bits:

    std::vector<float> fTrace(trace.begin(), trace.end());

    // Create torch tensor:
    
    auto opt = torch::TensorOptions().dtype(torch::kFloat32);
    TT input = torch::from_blob(fTrace.data(),
				{static_cast<long>(fTrace.size())},
				opt).clone();
    
    // Remove the offset:

    TT slice = std::get<0>(torch::topk(input, nsamples,
				       /*dim=*/0, /*largest=*/false,
				       /*sorted=*/false));
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
postProcessOutput(FitInfo* pResult, c10::IValue& output,
		  double xscale, TT yscale, TT offset)
{
    // Classification probability. First element is single pulse prob,
    // second element is double pulse prob:

    auto tup = output.toTuple();
    TT logit = tup->elements()[1].toTensor();
    TT pclass = torch::argmax(logit, 1); // 0: single pulse, 1: is double pulse
    TT prob = torch::softmax(logit, 1);

    // Single pulse results:

    TT t_s  = xscale * tup->elements()[2].toTensor();
    TT a_s  = yscale * tup->elements()[3].toTensor();
    TT k1_s = tup->elements()[4].toTensor() / xscale;
    TT k2_s = tup->elements()[5].toTensor() / xscale;
    //TT c_s  = yscale * tup->elements()[6].toTensor() + offset;
    
    // Double pulse results:

    TT t1_d = xscale * tup->elements()[12].toTensor();
    TT a1_d = yscale * tup->elements()[13].toTensor();
    TT k1_d = tup->elements()[14].toTensor() / xscale;
    TT k2_d = tup->elements()[15].toTensor() / xscale;
    TT t2_d = xscale * tup->elements()[17].toTensor();
    TT a2_d = yscale * tup->elements()[18].toTensor();
    TT k3_d = tup->elements()[19].toTensor() / xscale;
    TT k4_d = tup->elements()[20].toTensor() / xscale;
    // TT c_d = yscale * (tup->elements()[16].toTensor() + tup->elements()[21].toTensor()) + offset;
    
    // Populate our result:
    
    pResult->s_extension.onePulseFit.pulse.amplitude = a_s.item<double>();
    pResult->s_extension.onePulseFit.pulse.position  = t_s.item<double>();
    pResult->s_extension.onePulseFit.pulse.decayTime = k1_s.item<double>();
    pResult->s_extension.onePulseFit.pulse.steepness = k2_s.item<double>();
    //pResult->s_extension.onePulseFit.offset = c_s.item<double>();
    pResult->s_extension.onePulseFit.offset = offset.item<double>();
    
    pResult->s_extension.twoPulseFit.pulses[0].amplitude = a1_d.item<double>();
    pResult->s_extension.twoPulseFit.pulses[0].position  = t1_d.item<double>();
    pResult->s_extension.twoPulseFit.pulses[0].decayTime = k1_d.item<double>();
    pResult->s_extension.twoPulseFit.pulses[0].steepness = k2_d.item<double>();
    pResult->s_extension.twoPulseFit.pulses[1].amplitude = a2_d.item<double>();
    pResult->s_extension.twoPulseFit.pulses[1].position  = t2_d.item<double>();
    pResult->s_extension.twoPulseFit.pulses[1].decayTime = k3_d.item<double>();
    pResult->s_extension.twoPulseFit.pulses[1].steepness = k4_d.item<double>();
    //pResult->s_extension.twoPulseFit.offset = c_d.item<double>();
    pResult->s_extension.twoPulseFit.offset = offset.item<double>();
    
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
 */
void
ddastoys::mlinference::performInference(
    FitInfo* pResult,
    std::vector<uint16_t>& trace, unsigned saturation,
    torch::jit::script::Module& module
    )
{
    // Preprocess, store normalization constants for later:

    unsigned baselineSamples = 15; // How many samples for baseline removal?
    auto preprocessed = preprocessInput(trace, baselineSamples);
    
    auto normalizedInput = std::get<0>(preprocessed); // torch::Tensor
    auto max = std::get<1>(preprocessed);             //      |
    auto offset = std::get<2>(preprocessed);          //      V
    auto samples = static_cast<double>(trace.size());

    normalizedInput = normalizedInput.unsqueeze(0); // trace.size() x 1 matrix
    
    // Inference step, input must be matrix:

    c10::IValue output = module.forward({normalizedInput});

    // Fish out and populate the results:

    postProcessOutput(pResult, output, samples, max, offset);
}
