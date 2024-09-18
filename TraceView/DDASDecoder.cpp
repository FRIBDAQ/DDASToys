/** 
 * @file DDASDecoder.cpp
 * @brief Implement class for processing events. Create and call a ring item 
 * processor and let it decide what to do with the data.
 */

#include "DDASDecoder.h"

#include <iostream>
#include <cstdlib>
#include <memory>
#include <cstdint>
#include <sstream>

#include <URL.h>
#include <CDataSource.h>                // Abstract source of ring items.
#include <CDataSourceFactory.h>         // Turn URI into a concrete data source
#include <CRingItem.h>                  // Base class for ring items.
#include <DataFormat.h>                 // Ring item data formats.
#include <Exception.h>                  // Base class for exception handling.
#include <CRingItemFactory.h>           // Creates ring item from generic.
#include <CRingScalerItem.h>            // Specific ring item classes.
#include <CRingStateChangeItem.h>       //              |
#include <CRingTextItem.h>              //              |
#include <CPhysicsEventItem.h>          //              |
#include <CRingPhysicsEventCountItem.h> //              |
#include <CGlomParameters.h>            //              |
#include <CDataFormatItem.h>            //          ----+----

#include <DDASFitHit.h>
#include "TraceViewProcessor.h"

namespace ddastoys {

//____________________________________________________________________________
    DDASDecoder::DDASDecoder() :
	m_pSourceURL(nullptr), m_pSource(nullptr),
	m_pProcessor(new TraceViewProcessor), m_count(0)
    {}

//____________________________________________________________________________
    DDASDecoder::~DDASDecoder()
    {
	delete m_pSourceURL;
	delete m_pSource;
	delete m_pProcessor;
    }

//____________________________________________________________________________
    void
    DDASDecoder::createDataSource(std::string src)
    {
	m_count = 0; // Reset when creating a new source
	src = "file://" + src; // File URI formatting.
	std::vector<std::uint16_t> sample;
	std::vector<std::uint16_t> exclude;
	m_pSourceURL = new URL(src);
	std::cout << "Filename in DDASDecoder: "
		  << m_pSourceURL->getPath() << std::endl;
  
	try {
	    m_pSource = CDataSourceFactory::makeSource(src, sample, exclude);
	}
	catch(CException& e) {
	    std::cerr << "Failed to open the data source " << src << ": "
		      << e.ReasonText() << std::endl;
	    std::exit(EXIT_FAILURE);
	}
    }

//____________________________________________________________________________
/**
 * @details
 * An event is a collection of DDASFitHits stored in a vector.
 */
    std::vector<DDASFitHit>
    DDASDecoder::getEvent()
    {
	// Get the next PHYSICS_EVENT
  
	CRingItem* pItem = getNextPhysicsEvent();
	std::unique_ptr<CRingItem> item(pItem);

	// If we've found a PHYSICS_EVENT return the list of unpacked hits
	// Else its the end of the file, return an empty vector.

	if (item) {
	    processRingItem(*item);   
	    return m_pProcessor->getUnpackedHits();
	} else {
	    std::vector<DDASFitHit> v;
	    return v;
	}
    }

//____________________________________________________________________________
/**
 * @details
 * Skip forward by nevts PHYSICS_EVENTS where nevts is provided as an argument.
 */
    int
    DDASDecoder::skip(int nevts)
    {
	CRingItem* pItem;
	while(nevts > 0) {
	    pItem = getNextPhysicsEvent(); // nullptr on end
    
	    if (!pItem) { return -1; }
    
	    delete pItem;
	    nevts--;
	}

	return 0;
    }

//____________________________________________________________________________
    std::string
    DDASDecoder::getFilePath()
    {
	if (!m_pSourceURL)
	    return "";
	else
	    return m_pSourceURL->getPath();
    }

///
// Private methods
//

//____________________________________________________________________________
/**
 * @details
 * Keep looping over events until either:
 *   - A PHYSICS_EVENT is encountred, in this case, return it.
 *   - EOF is reached.
 * 
 */
    CRingItem*
    DDASDecoder::getNextPhysicsEvent()
    {
	CRingItem* pItem;
	while(true) {    
	    pItem = m_pSource->getItem(); // nullptr on end

	    if (!pItem) {
		return pItem;
	    } else if (pItem->type() == PHYSICS_EVENT) {
		m_count++;
		return pItem;
	    }
    
	    delete pItem;
	}
    }

//____________________________________________________________________________
/**
 * @details
 * The processor will handle specifically what to do with each ring item.
 */
    void
    DDASDecoder::processRingItem(CRingItem& item)
    {    
	CRingItem* castableItem = CRingItemFactory::createRingItem(item);
	std::unique_ptr<CRingItem> autoDeletedItem(castableItem);
    
	// Depending on the ring item type dynamic_cast the ring item to the
	// appropriate final class and invoke the correct handler. The default
	// case just invokes the unknown item type handler.
    
	switch (castableItem->type()) {
	case PERIODIC_SCALERS:
	{    
	    CRingScalerItem& scaler(
		dynamic_cast<CRingScalerItem&>(*castableItem)
		);
	    m_pProcessor->processScalerItem(scaler);
	    break;
	}
	case BEGIN_RUN: // All of these are state changes
	case END_RUN:
	case PAUSE_RUN:
	case RESUME_RUN:
	{
	    CRingStateChangeItem& statechange(
		dynamic_cast<CRingStateChangeItem&>(*castableItem)
		);
	    m_pProcessor->processStateChangeItem(statechange);
	    break;
	}
	case PACKET_TYPES: // Both are textual item types
	case MONITORED_VARIABLES:
	{
	    CRingTextItem& text(dynamic_cast<CRingTextItem&>(*castableItem));
	    m_pProcessor->processTextItem(text);
	    break;
	}
	case PHYSICS_EVENT:
	{
	    CPhysicsEventItem& event(
		dynamic_cast<CPhysicsEventItem&>(*castableItem)
		);
	    m_pProcessor->processEvent(event);
	    break;
	}
	case PHYSICS_EVENT_COUNT:
	{
	    CRingPhysicsEventCountItem&	eventcount(
		dynamic_cast<CRingPhysicsEventCountItem&>(*castableItem)
		);
	    m_pProcessor->processEventCount(eventcount);
	    break;
	}
	case RING_FORMAT:
	{
	    CDataFormatItem& format(
		dynamic_cast<CDataFormatItem&>(*castableItem)
		);
	    m_pProcessor->processFormat(format);
	    break;
	}
	case EVB_GLOM_INFO:
	{
	    CGlomParameters& glomparams(
		dynamic_cast<CGlomParameters&>(*castableItem)
		);
	    m_pProcessor->processGlomParams(glomparams);
	    break;
	}
	default:
	{
	    m_pProcessor->processUnknownItemType(item);
	    break;
	}
	}
    }

}
