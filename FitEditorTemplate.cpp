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

    // We also have to read the template file
    try {
	m_pConfig->readTemplateFile();
    }
    catch (std::exception& e) {
	std::cerr << "Error configuring FitEditor: " << e.what() << std::endl;
	exit(EXIT_FAILURE);
    }
}

/**
 * @brief Copy constructor.
 *
 * @param rhs Object to copy construct.
 */
FitEditorTemplate::FitEditorTemplate(const FitEditorTemplate& rhs) :
    m_pConfig(new Configuration(*rhs.m_pConfig))
{}

/**
 * @brief Move constructor.
 *
 * @param rhs Object to move construct.
 *
 * @details
 * Constructs using move assignment.
 */
FitEditorTemplate::FitEditorTemplate(FitEditorTemplate&& rhs) noexcept :
    m_pConfig(nullptr)
{
    *this = std::move(rhs);
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
 * @brief Copy assignment operator.
 *
 * @param rhs Object to copy assign.
 *
 * @return Reference to created object.
 */
FitEditorTemplate&
FitEditorTemplate::operator=(const FitEditorTemplate& rhs)
{
    if (this != &rhs) {
	delete m_pConfig;
	m_pConfig = new Configuration(*rhs.m_pConfig);
    }

    return *this;
}

/**
 * @brief Move assignment operator.
 *
 * @param rhs Object to move assign.
 *
 * @return Reference to created object.
 */
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
    // std::uint32_t of the body is the size of the standard hit part in
    // std::uint16_t words.
    
    std::uint16_t* pSize = static_cast<std::uint16_t*>(pBody);
    CBuiltRingItemEditor::BodySegment hitInfo(
	*pSize*sizeof(std::uint16_t),pSize, false
	);
    result.push_back(hitInfo);
    
    // Make the hit:    
    DAQ::DDAS::DDASHit hit;
    DAQ::DDAS::DDASHitUnpacker unpacker;
    unpacker.unpack(
	static_cast<std::uint32_t*>(pBody),
	static_cast<std::uint32_t*>(nullptr),
	hit
	);

    unsigned crate = hit.getCrateID();
    unsigned slot  = hit.getSlotID();
    unsigned chan  = hit.getChannelID();
  
    if (m_pConfig->fitChannel(crate, slot, chan)) {
	std::vector<std::uint16_t> trace = hit.getTrace();

	FitInfo* pFit = new FitInfo; // Have an extension though may be zero 
    
	if (trace.size() > 0) { // Need a trace to fit
	    std::pair<std::pair<unsigned, unsigned>, unsigned> l
		= m_pConfig->getFitLimits(crate, slot, chan);
	    unsigned low = l.first.first;   // Left fit limit
	    unsigned hi  = l.first.second;  // Right fit limit
	    unsigned sat = l.second;        // Saturation value

	    /** 
	     * @todo (ASC 2/6/23): Trace template as a temporary variable is 
	     * inefficient, though the fitting is what really takes time. 
	     * Still, get once, use many times.
	     */      
	    std::vector<double> traceTemplate = m_pConfig->getTemplate();
	    unsigned align = m_pConfig->getTemplateAlignPoint();
      
	    if (low != hi) {	
		int classification = pulseCount(hit);	
		if (classification) {	  
		    // Bit 0 do single fit.
		    // Bit 1 do double fit.                    
		    if (classification & 1) {
			DDAS::TemplateFit::lmfit1(
			    &(pFit->s_extension.onePulseFit), trace,
			    traceTemplate, align, l.first, sat
			    );
		    }
                    
		    if (classification & 2 ) {
			// Single pulse fit guides initial guess for double
			// pulse. If the single pulse fit does not exist, we
			// do it here.
			DDAS::fit1Info guess;
			if ((classification & 1) == 0) {
			    DDAS::TemplateFit::lmfit1(
				&(pFit->s_extension.onePulseFit), trace,
				traceTemplate, align, l.first, sat
				);
			} else {
			    guess = pFit->s_extension.onePulseFit;
			}
			DDAS::TemplateFit::lmfit2(
			    &(pFit->s_extension.twoPulseFit), trace,
			    traceTemplate, align, l.first, &guess, sat
			    );
		    }	  
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
FitEditorTemplate::pulseCount(DAQ::DDAS::DDASHit& hit)
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
    FitEditorTemplate* createEditor() {
	return new FitEditorTemplate;
    }
}
