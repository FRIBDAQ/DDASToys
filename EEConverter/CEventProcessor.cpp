/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins 
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  CEventProcessor.cpp
 * @brief Implement class for processing events. Create and call a ring item 
 * processor and let it decide what to do with the data.
 */

#include "CEventProcessor.h"

#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>
#include <cstdint>
#include <sstream>

#include <URL.h>
#include <CDataSource.h>                  // Abstract source of ring items
#include <CDataSourceFactory.h>           // URI to a concrete data source
#include <CRingItem.h>                    // Base class for ring items
#include <DataFormat.h>                   // Ring item data formats
#include <Exception.h>                    // Base class for exception handling
#include <CRingItemFactory.h>             // Create specific item from generic
#include <CRingScalerItem.h>              // Specific ring item classes
#include <CRingStateChangeItem.h>         //                 |
#include <CRingTextItem.h>                //                 |
#include <CPhysicsEventItem.h>            //                 |
#include <CRingPhysicsEventCountItem.h>   //                 |
#include <CDataFormatItem.h>              //                 |
#include <CGlomParameters.h>              //                 |
#include <CDataFormatItem.h>              //             ----+----

#include "CRingItemProcessor.h"
#include "ProcessToRootSink.h"
#include "StringsToIntegers.h"
#include "converterargs.h"

//____________________________________________________________________________
/**
 * @details 
 * No real action occurs until the operator() is called, as all of the 
 * interesting data must be determined by parsing the command line arguments.
 */
CEventProcessor::CEventProcessor() :
    m_pDataSource(nullptr), m_pProcessor(nullptr),
    m_itemCount(0), m_skipCount(0)
{}

//____________________________________________________________________________
CEventProcessor::~CEventProcessor()
{
    delete m_pDataSource;
    delete m_pProcessor;
}

//____________________________________________________________________________
/**
 * @details
 * Here we:
 *   - Parse arguments,
 *   - Open the data source,
 *   - Read data event-by-event,
 *   - Invoke a processor to process the individual ring items.
 */
int
CEventProcessor::operator()(int argc, char* argv[])
{
    // Parse arguments:
  
    gengetopt_args_info parse;
    cmdline_parser(argc, argv, &parse);  
  
    // Create the data source. Data sources allow us to specify ring item
    // types that will be skipped. They also allow us to specify types
    // that we may only want to sample (e.g. for online ring items).
  
    std::vector<std::uint16_t> sample; // Not used
    std::vector<std::uint16_t> exclude;

    /**
     * @todo (ASC 1/20/23): Awkward way to process the sample and exclude 
     * types. Can we return a vector of std::uint16_t or whatever the 
     * CDataSourceFactory wants? Item type codes are std::uint32_t so why are
     * we mixing between signed/unsigned ints and different bits?
     */
  
    std::vector<int> excludeToInt;
    if (parse.exclude_given) {
	try {
	    excludeToInt = stringListToIntegers(std::string(parse.exclude_arg));
	}
	catch (CException& e) {
	    std::cerr << "Invalid value for --exclude, must be a list of item types but was: " << std::string(parse.exclude_arg) << std::endl;
	    throw;    
	}
	for (const auto& ele : excludeToInt) {
	    exclude.push_back(ele);
	}
    }

    // Now we can actually make the source:

    /**
     * @todo (ASC 3/10/23): To support different NSCLDAQ data formats, we need 
     * to read the version string from the args and create an appropriate ring 
     * item factory. The ROOT converter should be independent of the data 
     * format since we can hand it CPhysicsEventItems from the correct format 
     * type. 
     */
  
    std::string sourceName;
    if (parse.source_given) {
	sourceName = parse.source_arg;
    }
  
    m_pDataSource = new URL(sourceName);
    std::cout << "Filename in CEventProcessor: "
	      << m_pDataSource->getPath() << std::endl;
  
    CDataSource* pSource = nullptr;
    try {
	pSource = CDataSourceFactory::makeSource(sourceName, sample, exclude);
    }
    catch(CException& e) {
	std::cerr << "Failed to open the data source " << sourceName << ": "
		  << e.ReasonText() << std::endl;
	throw;
    }
    std::unique_ptr<CDataSource> source(pSource);

    // Make the processor using the outout data format and file sink name:
  
    std::string sink;
    if (parse.fileout_given) {
	sink = parse.fileout_arg;
    }  

    // Can use the factory method and the outfmt arg to select a processor
    // but for now we know its a basic ROOT sink so just make one of those:

    m_pProcessor = new ProcessToRootSink(sink);

    // After we set the skip and item count we can begin reading data:

    if (parse.skip_given) {
	if (parse.skip_arg < 0) {
	    std::stringstream msg;
	    msg << "--skip value must be >= 0 but is "
		<< parse.skip_arg << std::endl;
	    throw std::invalid_argument(msg.str());
	} else {
	    m_skipCount = parse.skip_arg;
	}
    }
  
    if (parse.count_given) {
	if (parse.count_arg < 0) {
	    std::stringstream msg;
	    msg << "--count value must be >= 0 but is "
		<< parse.count_arg << std::endl;
	    throw std::invalid_argument(msg.str());
	} else {
	    m_itemCount = parse.count_arg;
	}
    }
  
    // The loops below consume items from the ring buffer until all are used up
    // or the item count limit is reached. The use of std::unique_ptrs ensure
    // that the dynamically created ring items we get from the data source are
    // automatically deleted when we exit the block in which it's created.
 
    // First we skip off the front of the event:
  
    size_t remain = m_skipCount;
    while (remain > 0) {
	std::unique_ptr<CRingItem> item(source->getItem());
	remain--;
    }

    // Then we get the items and process them:
  
    remain = m_itemCount;
    std::cout << "Count is " << m_itemCount << std::endl;
    bool done = false;
    while (!done) {
	std::unique_ptr<CRingItem> item(source->getItem());
	if (item.get()) {
	    processRingItem(*item);
	    remain--;    
	    if ((m_itemCount != 0) && (remain == 0)) {
		done = true;
	    }
	} else {
	    done = true; // No more items
	}
    }

    // Fall through to here when the source has no more data or we've reached
    // the item processing limit

    return EXIT_SUCCESS;  
}

//____________________________________________________________________________
/**
 * @brief Type-independent processing of ring items. The processor will handle 
 * specifically what to do with each ring item.
 *
 * @param item  References the ring item we got.
 */
void
CEventProcessor::processRingItem(CRingItem& item)
{
    // Create a dynamic ring item that can be dynamic cast to a specific one:
    
    CRingItem* castableItem = CRingItemFactory::createRingItem(item);
    std::unique_ptr<CRingItem> autoDeletedItem(castableItem);
    
    // Depending on the ring item type dynamic_cast the ring item to the
    // appropriate final class and invoke the correct handler. The default
    // case just invokes the unknown item type handler.
    
    switch (castableItem->type()) {
    case PERIODIC_SCALERS:
    {    
	CRingScalerItem& scaler(dynamic_cast<CRingScalerItem&>(*castableItem));
	m_pProcessor->processScalerItem(scaler);
	break;
    }
    case BEGIN_RUN: // All of these are state changes
    case END_RUN:
    case PAUSE_RUN:
    case RESUME_RUN:
    {
	CRingStateChangeItem& statechange(dynamic_cast<CRingStateChangeItem&>(*castableItem));
	m_pProcessor->processStateChangeItem(statechange);
	break;
    }
    case PACKET_TYPES:        // Both are textual item types
    case MONITORED_VARIABLES:
    {
	CRingTextItem& text(dynamic_cast<CRingTextItem&>(*castableItem));
	m_pProcessor->processTextItem(text);
	break;
    }
    case PHYSICS_EVENT:
    {
	CPhysicsEventItem& event(dynamic_cast<CPhysicsEventItem&>(*castableItem));
	m_pProcessor->processEvent(event);
	break;
    }
    case PHYSICS_EVENT_COUNT:
    {
	CRingPhysicsEventCountItem&
	    eventcount(dynamic_cast<CRingPhysicsEventCountItem&>(*castableItem));
	m_pProcessor->processEventCount(eventcount);
	break;
    }
    case RING_FORMAT:
    {
	CDataFormatItem& format(dynamic_cast<CDataFormatItem&>(*castableItem));
	m_pProcessor->processFormat(format);
	break;
    }
    case EVB_GLOM_INFO:
    {
	CGlomParameters& glomparams(dynamic_cast<CGlomParameters&>(*castableItem));
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
