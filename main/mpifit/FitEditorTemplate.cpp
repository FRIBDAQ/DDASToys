/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/lice nses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     Aaron Chester
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  FitEditorTemplate.cpp
 *  @brief: FitEditor class for analytic fitting.
 */

#include "FitEditorTemplate.h"

#include <iostream>
#include <fstream>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "lmfit_template.h"

using namespace DDAS::TemplateFit;

/**
 * Construtor
 *   Read the template fit configuration file upon creation
 */
FitEditorTemplate::FitEditorTemplate()
{
  // For the template fit we also need to read in the template data. We read a
  // file pointed to by the environment variable TEMPLATE_CONFIGFILE similar
  // to how the fitting configuration is read.
  
  try {
    std::string name = getTemplateFilename("TEMPLATE_CONFIGFILE");
    readTemplateFile(name.c_str());
  }
  catch (std::exception& e) {
    std::cerr << "Error processing template configuration file: "
	      << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * Destructor
 */
FitEditorTemplate::~FitEditorTemplate()
{}

/**
 * operator()
 *    - Parse the fragment into a hit.
 *    - Produce a IOvec element for the existing hit (without any fit
 *      that migh thave been there).
 *    - See if the predicate says we should fit.
 *    - If so, create the trae.
 *    - Get the fit limits, and saturation.
 *    - Get the number of pulses to fit.
 *    - Do the fits.
 *    - Create an Iovec entry for the extension we created (dynamic).
 *
 * @param pHdr - Pointer to the ring item header of the hit.
 * @param bhdr - Pointer to the body header pointer for the hit.
 * @param bodySize - Number of _bytes_ in the body.
 * @param pBody    - Pointer to the body.
 * @return std::vector<CBuiltRingItemEditor::BodySegment>
 *             - final segment descriptors.
 */
std::vector<CBuiltRingItemEditor::BodySegment>
FitEditorTemplate::operator()(pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody)
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
    
  if (doFit(hit)) {
    std::vector<std::uint16_t> trace = hit.GetTrace();

    // \TODO (ASC 1/18/23): new'd without a delete, what happens?  
    pFitInfo pFit = new FitInfo; // Have an extension tho may be zero
    
    if (trace.size() > 0) { // Need a trace to fit
      auto l = fitLimits(hit);
      unsigned low = l.first.first;   // Left fit limit
      unsigned hi  = l.first.second;  // Right fit limit
      unsigned sat = l.second;        // Saturation value
           
      if (low != hi) {	
	int classification = pulseCount(hit);
	
	if (classification) {
	  
	  // Bit 0 do single fit.
	  // Bit 1 do double fit.
                    
	  if (classification & 1) {
	    lmfit1(&(pFit->s_extension.onePulseFit), trace, m_template,
		   m_alignPoint, l.first, sat);
	  }
                    
	  if (classification & 2 ) {
	    // Single pulse fit guides initial guess for double pulse. If the
	    // single pulse fit does not exist, we do it here.
	    DDAS::fit1Info guess;                    

	    if ((classification & 1) == 0) {
	      lmfit1(&(pFit->s_extension.onePulseFit), trace, m_template,
		     m_alignPoint, l.first, sat);
	    } else {
	      guess = pFit->s_extension.onePulseFit;
	    }
	    lmfit2(&(pFit->s_extension.twoPulseFit), trace, m_template,
		   m_alignPoint, l.first, &guess, sat);
	  }	  
	}	
      }      
    }
    
    CBuiltRingItemEditor::BodySegment fit(sizeof(FitInfo), pFit, true);
    result.push_back(fit);
    
  } else { // No fit performed
    pNullExtension p = new nullExtension;
    CBuiltRingItemEditor::BodySegment nofit(sizeof(nullExtension), p, true);
    result.push_back(nofit);
  }    
    
  return result; // Return the description
}

/**
 * free
 *   We get handed our fit extension descriptor(s) to free
 *
 * @param e - iovec we need to free
 */
void
FitEditorTemplate::free(iovec& e)
{
  if (e.iov_len == sizeof(FitInfo)) {
    pFitInfo pFit = static_cast<pFitInfo>(e.iov_base);
    delete pFit;
  } else {
    pNullExtension p = static_cast<pNullExtension>(e.iov_base);
    delete p;
  }
}

//
// Private methods
//


/**
 * getTemplateFilename
 *   Get template configuratin file name pointed to by an environment variable
 *
 * @param envname - the name of the environment variable pointing to the 
 *                  configuration file
 *
 * @return std::string - the filename
 *
 * @throw std::logic_error - if the envname does not point to a file
 */
std::string
FitEditorTemplate::getTemplateFilename(const char* envname)
{
    const char* pFilename = getenv(envname);
    if (!pFilename) {
        std::string msg("No translation for environment variable: ");
        msg += envname;
        msg += " Point that to the template configuration file and re-run";
        throw std::invalid_argument(msg);
    }
    
    return std::string(pFilename);
}

/**
 * readTemplateFile
 *   Read the formatted tempalate configuration in formation and template data 
 *   from a file.
 *
 * @param filename - file name to read config info from
 *
 * @throw std::length_error - if the number of template data points is 
 *                            different than what the configuration file expects
 * @throw std::invalid_arugment - if the alignment point of the template is 
 *                                not contained in the trace (eg align to 
 *                                sample 101 on a 100 sample trace)
 * @throw std::invalid_argument - if the template data file cannot be opened
 */
void
FitEditorTemplate::readTemplateFile(const char* filename)
{
  std::ifstream fileIn;
  fileIn.open(filename, std::ifstream::in);
  if (fileIn.is_open()) {
    unsigned npts;
    double val;

    // \TODO (ASC 1/25/23): What happens when there are fewer than two values on the first line? Should report an error and stop trying to do the fit.
    fileIn >> m_alignPoint >> npts;    
    while (fileIn >> val) {
      m_template.push_back(val);
    }

    // The template should know how long it is. If you read in more data
    // points throw an exception.
    if (m_template.size() != npts) {
      std::string errmsg("Template configfile thinks the trace is ");
      errmsg += npts;
      errmsg += " samples but read in ";
      errmsg += m_template.size();
      throw std::length_error(errmsg); // I guess this is the right one?
    }

    // Ensure the alignment point is contained in the trace. Note that because
    // m_alignPoint is an unsigned type it cannot be negative.
    if (m_alignPoint >= m_template.size()) {
      std::string errmsg("Invalid template alignment point ");
      errmsg += m_alignPoint;
      errmsg += " >= template size ";
      errmsg += m_template.size();
      throw std::invalid_argument(errmsg);
    }
    
  } else {
    std::string errmsg("Cannot open template data file: ");
    errmsg += filename;
    throw std::invalid_argument(errmsg);
  }

  fileIn.close();
}

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

/*
 * doFit
 *   This is a predicate function:
 *
 * @param crate - crate a hit comes from.
 * @param slot  - slot a hit comes from.
 * @param channel - channel within the slot the hit comes from
 *
 * @return bool - if true, the channel is fit.
 *
 * @note This sample predicate requests that all channels be fit.
 */
bool
FitEditorTemplate::doFit(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();
    int chan  = hit.GetChannelID();    
    int index = channelIndex(crate, slot, chan);
    
    return (m_fitChannels.find(index) != m_fitChannels.end());
}

/**
* fitLimits
*   For each channel we're fitting, we need to know
*   - The left and right limits of the waveform that will be fitted
*   - The digitizer saturation level
*
*
* @param hit - reference to a single hit
*
* @return std::pair<std::pair<unsigned, unsigned>, unsigned>
*   The first element of the outer pair are the left and right
*   fit limits respectively. The second element of the outer pair
*   is the saturation level.
*
* @note the caller must have ensured there's a map entry for this channel.
*/
std::pair<std::pair<unsigned, unsigned>, unsigned>
FitEditorTemplate::fitLimits(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();
    int chan  = hit.GetChannelID();    
    int index = channelIndex(crate, slot, chan);
    
    std::pair<std::pair<unsigned, unsigned>, unsigned> result = m_fitChannels[index];
    
    return result;       
}

/////////////////////////////////////////////////////////////////////////////
// Factory for our editor:
//
extern "C" {
  FitEditorTemplate* createEditor() {
    return new FitEditorTemplate;
  }
}
