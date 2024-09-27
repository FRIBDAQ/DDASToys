/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
	     Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  FitEditorMLInference.cpp
 * @brief Implementation of the FitEditor class for machine-learning inference.
 */

#include "FitEditorMLInference.h"

#include <iostream>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "Configuration.h"
#include "fit_extensions.h"
#include "mlinference.h"

using namespace ddasfmt;
using namespace ddastoys;

/**
 * @details
 * Sets up the configuration manager to parse config files and manage 
 * configuration data. Reads the fit config file. Loads all ML models
 * specified in the configuration. Stores models in a map keyed by the
 * path to their associated PyTorch file.
 */
ddastoys::FitEditorMLInference::FitEditorMLInference() :
    m_pConfig(new Configuration)
{  
    try {
	m_pConfig->readConfigFile();
    }
    catch (std::exception& e) {
	std::cerr << "Error configuring FitEditor: " << e.what() << std::endl;
	exit(EXIT_FAILURE);
    }

    // Load all the models and store for later access:
    
    auto modelList = m_pConfig->getModelList();
    for (const auto& m : modelList) {
	try {
	    torch::jit::script::Module module = torch::jit::load(m);
	    m_models[m] = module;
	}
	catch (const c10::Error& e) {
	    std::cerr << "Failed to load model " << m << ": " << e.what()
		      << std::endl;
	    exit(EXIT_FAILURE);
	}
    }
}

ddastoys::FitEditorMLInference::FitEditorMLInference(
    const FitEditorMLInference& rhs
    ) :
    m_pConfig(new Configuration(*rhs.m_pConfig))
{}

/**
 * @details
 * Constructs using move assignment.
 */
ddastoys::FitEditorMLInference::FitEditorMLInference(
    FitEditorMLInference&& rhs
    ) noexcept :
    m_pConfig(nullptr)
{
    *this = std::move(rhs);
}

FitEditorMLInference&
ddastoys::FitEditorMLInference::operator=(const FitEditorMLInference& rhs)
{
    if (this != &rhs) {
	delete m_pConfig;
	m_pConfig = new Configuration(*rhs.m_pConfig);
    }

    return *this;
}

FitEditorMLInference&
ddastoys::FitEditorMLInference::operator=(FitEditorMLInference&& rhs) noexcept
{
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
ddastoys::FitEditorMLInference::~FitEditorMLInference()
{
    delete m_pConfig;
}

/**
 * @details
 * This is the hook into the FitEditorMLInference class. Here we:
 * - Parse the fragment into a hit.
 * - Produce a IOvec element for the existing hit (without any fit
 *   that might have been there).
 * - See if the configuration manager says we should fit and if so, get the 
 *   trace from the hit.
 * - Do the inference step
 * - Create an IOvec entry for the extension we created (dynamic).
 */
std::vector<CBuiltRingItemEditor::BodySegment>
ddastoys::FitEditorMLInference::operator()(
    pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize, void* pBody
    )
{ 
    std::vector<CBuiltRingItemEditor::BodySegment> result;
    
    // Regardless we want a segment that includes the hit. Note that the first
    // uint32_t of the body is the size of the standard hit part in
    // uint16_t words.
    
    uint16_t* pSize = static_cast<uint16_t*>(pBody);
    CBuiltRingItemEditor::BodySegment hitInfo(
	*pSize*sizeof(uint16_t), pSize, false
	);
    result.push_back(hitInfo);
    
    // Make the hit:
    
    DDASHit hit;
    DDASHitUnpacker unpacker;
    unpacker.unpack(
	static_cast<uint32_t*>(pBody),
	static_cast<uint32_t*>(nullptr),
	hit
	);

    auto crate = hit.getCrateID();
    auto slot  = hit.getSlotID();
    auto chan  = hit.getChannelID();
  
    if (m_pConfig->fitChannel(crate, slot, chan)) {
	auto trace = hit.getTrace();
	FitInfo* pFit = new FitInfo; // Have an extension tho may be zero.
	
	if (trace.size() > 0) { // Need a trace to fit
	    auto sat = m_pConfig->getSaturationValue(crate, slot, chan);
	    auto modelPath = m_pConfig->getModelPath(crate, slot, chan);
	    mlinference::performInference(
		pFit, trace, sat, m_models[modelPath]
		);
	}
    
	CBuiltRingItemEditor::BodySegment fit(sizeof(FitInfo), pFit, true);
	result.push_back(fit);
    
    } else { // No fit performed
	nullExtension* p = new nullExtension;
	CBuiltRingItemEditor::BodySegment nofit(sizeof(nullExtension), p, true);
	result.push_back(nofit);
    }
    
    return result; // Return the description
}

void
ddastoys::FitEditorMLInference::free(iovec& e)
{
    if (e.iov_len == sizeof(FitInfo)) {
	FitInfo* pFit = static_cast<FitInfo*>(e.iov_base);
	delete pFit;
    } else {
	nullExtension* p = static_cast<nullExtension*>(e.iov_base);
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
    ddastoys::FitEditorMLInference* createEditor() {
	return new ddastoys::FitEditorMLInference;
    }
}
