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
 * @brief Function implementations for machine learning inference editor.
 */

#include "mlinference.h"

#include <algorithm>

#include <torch/torch.h>

#include "fit_extensions.h"

using namespace ddastoys;

/**
 * @todo (ASC 9/19/24): Does using input/output parameters for pre- and 
 * post-processing data improve performance (avoid multiple 
 * construciton/assignment)?
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
 * @note It is currently unclear whether this is needed. The infernce model is 
 * trained using all samples of unsaturated traces. Reducing the trace changes 
 * the input shape. What is the model performance on saturated traces?
 */
static std::vector<std::pair<uint16_t, uint16_t>>
reduceTrace(std::vector<uint16_t>& trace, unsigned saturation)
{
    std::vector<std::pair<uint16_t, uint16_t>> reduced;
    for (int i = 0; i < trace.size(); i++) {
	if (trace[i] < saturation) {
	    reduced.push_back(std::pair<uint16_t, uint16_t>(i, trace[i]));
	}
    }
    
    return reduced;
}


/*------------------------------------------------------------------
 * Input and output data processing.
 */

/**
 * @brief Pre-process the input and return a torch tensor.
 * 
 * @param trace References the trace to process.
 * @param nsamples Number of samples for baseline estimation to remove offset. 
 *
 * @return Pre-processed data as a PyTorch tensor.
 *
 * @details
 * Pre-processing involves two steps:
 * - Remove the DC offset.
 * - Normalize the trace by dividing each sample by the max value.
 * The trace-as-a-normalized-tensor is floating point.
 *
 * @note From https://github.com/bashir-sadeghi/frib/tree/main/pytorch_to_cpp
 * @todo (ASC 9/19/24): The number of samples to estimate the baseline must be 
 * by default smaller to account for short trigger delays or preferrably 
 * configurable by the user.
 */
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
preprocessInput(std::vector<uint16_t>& trace, unsigned nsamples)
{    
    // Convert the trace into a torch tensor. Note that first we transform
    // the input to float and hope this is 32 bits...

    std::vector<float> input(trace.begin(), trace.end());

    // Create torch tensor:
    
    auto opt = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor inputTensor = torch::from_blob(
	input.data(), {static_cast<long>(trace.size())}, opt
	).clone();
    
    // Remove the offset:

    torch::Tensor slice = inputTensor.narrow(0, 0, nsamples);
    torch::Tensor offset = torch::mean(slice);
    auto normInputTensor = inputTensor - offset;
  
    // Divide each row by its maximum value:

    torch::Tensor max = torch::max(inputTensor);
    normInputTensor /= max;
    
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
	normInputTensor, max, offset
	);
}

/**
 * @brief Post-process the input and return the results.
 *
 * @param pResult Pointer to results storage.
 * @param output References our processed tensor.
 * @param xscale x-axis normalziation factor (the trace size).
 * @param yscale y-axis normalization factor (max value)
 *
 * @return Fit struct containing the results.
 */
static void
postProcessOutput(
    FitInfo* pResult, c10::IValue& output,
    double xscale, torch::Tensor yscale, torch::Tensor offset
    )
{
    // Classification probability. First element is single pulse prob,
    // second element is double pulse prob:
    
    auto tup = output.toTuple();
    torch::Tensor logit = tup->elements()[1].toTensor();
    torch::Tensor prob = torch::softmax(logit, 1); 

    // // Extract the parameters and rescale them:
    
    torch::Tensor t1 = xscale * tup->elements()[2].toTensor();
    torch::Tensor a1 = yscale * tup->elements()[3].toTensor();
    torch::Tensor k1 = tup->elements()[4].toTensor() / xscale;
    torch::Tensor k2 = tup->elements()[5].toTensor() / xscale;
    // torch::Tensor c1 = xscale * tup->elements()[6].toTensor();
    torch::Tensor t2 = xscale * tup->elements()[7].toTensor();
    torch::Tensor a2 = yscale * tup->elements()[8].toTensor();
    torch::Tensor k3 = tup->elements()[9].toTensor() / xscale;
    torch::Tensor k4 = tup->elements()[10].toTensor() / xscale;
    torch::Tensor c = yscale * tup->elements()[11].toTensor() + offset;

    // // Populate our result and return it:
    
    pResult->s_extension.twoPulseFit.pulses[0].amplitude = a1.item<double>();
    pResult->s_extension.twoPulseFit.pulses[0].position = t1.item<double>();
    pResult->s_extension.twoPulseFit.pulses[0].decayTime = k1.item<double>();
    pResult->s_extension.twoPulseFit.pulses[0].steepness = k2.item<double>();

    pResult->s_extension.twoPulseFit.pulses[1].amplitude = a2.item<double>();
    pResult->s_extension.twoPulseFit.pulses[1].position = t2.item<double>();
    pResult->s_extension.twoPulseFit.pulses[1].decayTime = k3.item<double>();
    pResult->s_extension.twoPulseFit.pulses[1].steepness = k4.item<double>();

    pResult->s_extension.twoPulseFit.offset = c.item<double>();
    
    // pResult->s_extension.singleProb = prob[0].item<double>();
    // pResult->s_extension.doubleProb = prob[1].item<double>();
}

/*------------------------------------------------------------------
 * Perform the inference:
 */

void
ddastoys::mlinference::performInference(
    FitInfo* pResult,
    std::vector<uint16_t>& trace, unsigned saturation,
    torch::jit::script::Module& module
    )
{
    // Preprocess, store normalization constants for later:

    unsigned baselineSamples = 20; // How many samples for baseline removal?
    auto preprocessed = preprocessInput(trace, baselineSamples);
    
    auto normalizedInput = std::get<0>(preprocessed);
    auto max = std::get<1>(preprocessed);
    auto offset = std::get<2>(preprocessed);
    auto samples = static_cast<double>(trace.size());

    normalizedInput = normalizedInput.unsqueeze(0); // trace.size() x 1 matrix
    
    // Inference step, input must be matrix:

    c10::IValue output = module.forward({normalizedInput});

    // Fish out and populate the results:

    postProcessOutput(pResult, output, samples, max, offset);
}
