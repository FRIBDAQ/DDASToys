/** 
 * @file  FitEditorTemplate.cpp
 * @brief Implementation of the FitEditor class for template fitting.
 */

/** @addtogroup TemplateFit
 * @brief Template FitEditor and fitting subroutines.
 * @{
 */

#include "FitEditorTemplate.h"

#include <iostream>
#include <fstream>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "Configuration.h"
#include "lmfit_template.h"

/**
 * @brief Constructor.
 * 
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
 * @brief Destructor.
 */
FitEditorTemplate::~FitEditorTemplate()
{
  delete m_pConfig;
}

/**
 * @brief Perform the fit and create a fit extension for a single event 
 *        fragment. 
 *
 * This is the hook into the FitEditorAnalytic class. Here we:
 * - Parse the fragment into a hit.
 * - Produce a IOvec element for the existing hit (without any fit
 *   that might have been there).
 * - See if the configuration manager says we should fit and if so, create 
 *   the trace.
 * - Get the fit limits and saturation value.
 * - Get the number of pulses to fit.
 * - Do the fits.
 * - Create an IOvec entry for the extension we created (dynamic).
 *
 * @param pHdr      Pointer to the ring item header of the hit.
 * @param pBHdr     Pointer to the body header pointer for the hit.
 * @param bodySize  Number of bytes in the body.
 * @param pBody     Pointer to the body.
 * 
 * @return std::vector<CBuiltRingItemEditor::BodySegment>  Final segment 
 *                                                         descriptors.
 */
std::vector<CBuiltRingItemEditor::BodySegment>
FitEditorTemplate::operator()(pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize, void* pBody)
{

  std::vector<CBuiltRingItemEditor::BodySegment> result;
    
  // Regardless we want a segment that includes the hit. Note that the first
  // std::uint32_t of the body is the size of the standard hit part in
  // std::uint16_t words.
    
  std::uint16_t* pSize = static_cast<std::uint16_t*>(pBody);
  CBuiltRingItemEditor::BodySegment hitInfo(*pSize*sizeof(std::uint16_t),
					    pSize, false);
  result.push_back(hitInfo);
    
  // Make the hit:
    
  DAQ::DDAS::DDASHit hit;
  DAQ::DDAS::DDASHitUnpacker unpacker;
  unpacker.unpack(static_cast<std::uint32_t*>(pBody),
		  static_cast<std::uint32_t*>(nullptr),
		  hit);

  unsigned crate = hit.GetCrateID();
  unsigned slot  = hit.GetSlotID();
  unsigned chan  = hit.GetChannelID();
  
  if (m_pConfig->fitChannel(crate, slot, chan)) {
    std::vector<std::uint16_t> trace = hit.GetTrace();

    FitInfo* pFit = new FitInfo; // Have an extension though may be zero 
    
    if (trace.size() > 0) { // Need a trace to fit
      std::pair<std::pair<unsigned, unsigned>, unsigned> l
	= m_pConfig->getFitLimits(crate, slot, chan);
      unsigned low = l.first.first;   // Left fit limit
      unsigned hi  = l.first.second;  // Right fit limit
      unsigned sat = l.second;        // Saturation value

      // \TODO (ASC 2/6/23): Making a copy for each hit we want to fit may be
      // inefficient, though probably (?) negligable compared to the fitting
      // itself. If we have an energy-dependendent template however, we may
      // want to get the proper template trace for this hit energy from some
      // large map in the Configuration.
      
      std::vector<double> traceTemplate = m_pConfig->getTemplate();
      unsigned align = m_pConfig->getTemplateAlignPoint();
      
      if (low != hi) {	
	int classification = pulseCount(hit);
	
	if (classification) {
	  
	  // Bit 0 do single fit.
	  // Bit 1 do double fit.
                    
	  if (classification & 1) {
	    DDAS::TemplateFit::lmfit1(&(pFit->s_extension.onePulseFit), trace, traceTemplate, align, l.first, sat);
	  }
                    
	  if (classification & 2 ) {
	    // Single pulse fit guides initial guess for double pulse. If the
	    // single pulse fit does not exist, we do it here.
	    DDAS::fit1Info guess;                    

	    if ((classification & 1) == 0) {
	      DDAS::TemplateFit::lmfit1(&(pFit->s_extension.onePulseFit), trace, traceTemplate, align, l.first, sat);
	    } else {
	      guess = pFit->s_extension.onePulseFit;
	    }
	    DDAS::TemplateFit::lmfit2(&(pFit->s_extension.twoPulseFit), trace,  traceTemplate, align, l.first, &guess, sat);
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

/**
 * @brief Free the dynamic fit extension descriptor(s).
 *
 * @param e  IOvec we need to free.
 */
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

//
// Private methods
//

/**
 * pulseCount
 *   This is a hook into which to add the ML classifier
 *
 * @param hit - references a hit.
 *
 * @return int
 * @retval 0  - On the basis of the trace no fitting.
 * @retval 1  - Only fit a single trace.
 * @retval 2  - Only fit two traces.
 * @retval 3  - Fit both one and double hit.
 */
int
FitEditorTemplate::pulseCount(DAQ::DDAS::DDASHit& hit)
{
    return 3;                  // in absence of classifier.
}

/////////////////////////////////////////////////////////////////////////////
// Factory for our editor:
//

/**
 * @brief Factory method to create this FitEditor.
 *
 * $DAQBIN/EventEditor expects a symbol called createEditor to exist in the 
 * plugin library it loads at runtime. Wrapping the factory method in 
 * extern "C" prevents namespace mangling by the C++ compiler.
 */
extern "C" {
  FitEditorTemplate* createEditor() {
    return new FitEditorTemplate;
  }
}

/** @} */
