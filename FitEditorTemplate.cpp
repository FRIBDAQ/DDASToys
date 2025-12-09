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
 * @file  FitEditorTemplate.cpp
 * @brief Implementation of the FitEditor class for template fitting.
 */

#include "FitEditorTemplate.h"

#include <iostream>
#include <fstream>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "Configuration.h"
#include "lmfit_template.h"
#include "profiling.h"

using namespace ddasfmt;
using namespace ddastoys;

static Stats stats;
    
/**
 * @details
 * Sets up the configuration manager to parse config files and manage 
 * configuration data. Read the fit configuration file and the template 
 * file on creation.
 */
FitEditorTemplate::FitEditorTemplate() :
    m_pConfig(new Configuration)
{
    try {
	m_pConfig->readConfigFile();
    }
    catch (std::exception& e) {
	std::cerr << "Error configuring FitEditor: " << e.what() << std::endl;
	exit(EXIT_FAILURE);
    }

    // We also have to read the template file:
    
    try {
	m_pConfig->readTemplateFile();
    }
    catch (std::exception& e) {
	std::cerr << "Error configuring FitEditor: " << e.what() << std::endl;
	exit(EXIT_FAILURE);
    }

    // Grab the template data and alignment point:

    /** 
     * @todo (ASC 9/17/24): Define a per-channel template. Requires getting 
     * the template for a crate/slot/channel combo rather than one template 
     * for the entire analysis. See comments in Configuration.h.
     */    
    m_template = m_pConfig->getTemplate();
    m_align = m_pConfig->getTemplateAlignPoint();
}

FitEditorTemplate::FitEditorTemplate(const FitEditorTemplate& rhs) :
    m_pConfig(new Configuration(*rhs.m_pConfig))
{}

/**
 * @details
 * Constructs using move assignment.
 */
FitEditorTemplate::FitEditorTemplate(FitEditorTemplate&& rhs) noexcept :
    m_pConfig(nullptr)
{
    *this = std::move(rhs);
}

FitEditorTemplate&
FitEditorTemplate::operator=(const FitEditorTemplate& rhs)
{
    if (this != &rhs) {
	delete m_pConfig;
	m_pConfig = new Configuration(*rhs.m_pConfig);
    }

    return *this;
}

FitEditorTemplate&
FitEditorTemplate::operator=(FitEditorTemplate&& rhs) noexcept
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
FitEditorTemplate::~FitEditorTemplate()
{
    delete m_pConfig;
}

/**
 * @details
 * This is the hook into the FitEditorTemplate class. Here we:
 * - Parse the fragment into a hit.
 * - Produce a IOvec element for the existing hit (without any fit
 *   that might have been there).
 * - See if the configuration manager says we should fit and if so, create 
 *   the trace.
 * - Get the fit limits and saturation value.
 * - Get the number of pulses to fit.
 * - Do the fits.
 * - Create an IOvec entry for the extension we created (dynamic).
 */
std::vector<CBuiltRingItemEditor::BodySegment>
FitEditorTemplate::operator()(
    pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize, void* pBody
    )
{
    std::vector<CBuiltRingItemEditor::BodySegment> result;
    
    // Regardless we want a segment that includes the hit. Note that the first
    // uint32_t of the body is the size of the standard hit part in
    // uint16_t words.
    
    uint32_t* pSize = static_cast<uint32_t*>(pBody);
    CBuiltRingItemEditor::BodySegment hitInfo(
	*pSize*sizeof(uint16_t),pSize, false
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
	std::vector<uint16_t> trace = hit.getTrace();
	FitInfo* pFit = new FitInfo; // Have an extension though may be zero 
    
	if (trace.size() > 0) { // Need a trace to fit
	    auto limits = m_pConfig->getFitLimits(crate, slot, chan);
	    auto sat = m_pConfig->getSaturationValue(crate, slot, chan);  
	    int classification = pulseCount(hit);
	    
	    if (classification) {

		// Track total time:

		double total = 0;
		
		// Bit 0 do single fit, bit 1 do double fit.
		
		if (classification & 1) {
		    Timer timer;
		    templatefit::lmfit1(
			&(pFit->s_extension.onePulseFit), trace,
			m_template, m_align, limits, sat
			);
		    total += timer.elapsed();
		}
                    
		if (classification & 2 ) {
		    
		    // The single pulse fit guides the double pulse fit.
		    // Note that lmfit2 will perform a single fit if no guess
		    // is provided. If we have already fit the single pulse,
		    // set the guess to those results.
		    
		    if (classification & 1) {
			fit1Info guess = pFit->s_extension.onePulseFit;
			Timer timer;
			templatefit::lmfit2(
			    &(pFit->s_extension.twoPulseFit), trace,
			    m_template, m_align, limits, &guess, sat
			    );
			total += timer.elapsed();
		    } else {
			// nullptr: no guess for single params.
			Timer timer;
			templatefit::lmfit2(
			    &(pFit->s_extension.twoPulseFit), trace,
			    m_template, m_align, limits, nullptr, sat
			    );
			total += timer.elapsed();
		    }
		}

		stats.addData(total);
		if (stats.size() == 10000) {
		    stats.compute();
		    stats.print("======== Template fit stats ========");
		}
		
	    }	   
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
FitEditorTemplate::free(iovec& e)
{
    if (e.iov_len == sizeof(FitInfo)) {
	FitInfo* pFit = static_cast<FitInfo*>(e.iov_base);
	delete pFit;
    } else {
	nullExtension* p = static_cast<nullExtension*>(e.iov_base);
	delete p;
    }
}

///
// Private methods
//

int
FitEditorTemplate::pulseCount(DDASHit& hit)
{
    return 3; // In absence of classifier.
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
    ddastoys::FitEditorTemplate* createEditor() {
	return new ddastoys::FitEditorTemplate;
    }
}
