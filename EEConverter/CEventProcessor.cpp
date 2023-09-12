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

/**
 * @todo (ASC 9/12/23): Currently can dump to stdout all data format items.
 * Still need to hook all this up to the ROOT sink. One known issue is that
 * the unified format library returns const uint16_t* to the body pointer
 * instead of uint16_t*, and the current FragmentIndexer cannot accept 
 * const uint16_t* body pointers.
 */

#include "CEventProcessor.h"

#include <iostream>
#include <cstdlib>
#include <memory>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>

#include <NSCLDAQFormatFactorySelector.h> // FRIBDAQ unified format includes
#include <RingItemFactoryBase.h>          //             ----+----

#include <URL.h>

// These are headers for the abstrct ring items we can get back from the
// factory. As new ring items are added this set of #include's must be updated
// as well as the switch statement in the processItem method.

#include <CRingItem.h>
#include <CAbnormalEndItem.h>
#include <CDataFormatItem.h>
#include <CGlomParameters.h>
#include <CPhysicsEventItem.h>
#include <CRingFragmentItem.h>
#include <CRingPhysicsEventCountItem.h>
#include <CRingScalerItem.h>
#include <CRingTextItem.h>
#include <CRingStateChangeItem.h>
#include <CUnknownFragment.h>

#include "CRingItemProcessor.h"
#include "RootSinkProcessor.h"
#include "DataSource.h"
#include "StreamDataSource.h"
#include "converterargs.h"

static std::map<std::string, uint32_t> TypeMap = {
    {"BEGIN_RUN", BEGIN_RUN},
    {"END_RUN", END_RUN},
    {"PAUSE_RUN", PAUSE_RUN},
    {"RESUME_RUN", RESUME_RUN},
    {"ABNORMAL_ENDRUN", ABNORMAL_ENDRUN},
    {"PACKET_TYPES", PACKET_TYPES},
    {"MONITORED_VARIABLES", MONITORED_VARIABLES},
    {"RING_FORMAT", RING_FORMAT},
    {"PERIODIC_SCALERS", PERIODIC_SCALERS},
    {"INCREMENTAL_SCALERS", INCREMENTAL_SCALERS},
    {"TIMESTAMPED_NONINCR_SCALERS", TIMESTAMPED_NONINCR_SCALERS},
    {"PHYSICS_EVENT", PHYSICS_EVENT},
    {"PHYSICS_EVENT_COUNT", PHYSICS_EVENT_COUNT},
    {"EVB_FRAGMENT", EVB_FRAGMENT},
    {"EVB_UNKNOWN_PAYLOAD", EVB_UNKNOWN_PAYLOAD},
    {"EVB_GLOM_INFO", EVB_GLOM_INFO}
};

//____________________________________________________________________________
/**
 * @brief Split the string using a delimiter.
 * 
 * @param str String to split up.
 * @param delim Delimimeter on which to split the string.
 *
 * @return std::vector<string> not that in the original, this is a parameter.
 *
 * @details
 * Shamelessly stolen from 
 * https://www.techiedelight.com/split-string-cpp-using-delimiter/
 */
static std::vector<std::string>
tokenize(std::string const &str, const char delim)
{
    std::vector<std::string> out;
    size_t start;
    size_t end = 0;
 
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
    
    return out;
}

//____________________________________________________________________________
/**
 * @brief Creates a vector of the ring item types to be excluded from the 
 * conversion given a comma separated list of types.
 *
 * @param exclusions String containing the exclusion list.
 *
 * @return std::vector<uint32_t> Items to exclude.
 *
 * @throw std::invalid_argument an exclusion item is not a string and is 
 *   not in the map of recognized item types.
 *
 * @details 
 * A type can be a string or a positive number. If it is a string, it is 
 * translated to the type id using TypeMap. If it is a number, use as is.
 */
static std::vector<uint32_t>
makeExclusionList(const std::string& exclusions)
{
    std::vector<uint32_t> result;
    std::vector<std::string> words = tokenize(exclusions, ',');
    
    // Process the words into an exclusion list:
    for (auto s : words) {
        bool isInt(true);
        int intValue;
        try {
            intValue = std::stoi(s);
        }
        catch (...) {
            // Failed as int.     
            isInt = false;
        }
        if (isInt) {
            result.push_back(intValue);
        } else {
            auto p = TypeMap.find(s);
            if (p != TypeMap.end()) {
                result.push_back(p->second);
            } else {
                std::string msg("Invalid item type in exclusion list: ");
                msg += s;
                throw std::invalid_argument(msg);
            }
        }
    }
    
    return result;
}

//____________________________________________________________________________
/**

 * @details Map the version we get from the command line to a factory version.
 *
 * @param fmtIn Format the user requested.
 *
 * @return Factory version ID.
 *
 * @throw std::invalid_argument Bad format version
 *
 * @note We should never throw because gengetopt will enforce the enum.
 */
static FormatSelector::SupportedVersions
mapVersion(enum_format fmtIn)
{
    switch (fmtIn) {
        case format_arg_v12:
            return FormatSelector::v12;
        case format_arg_v11:
            return FormatSelector::v11;
        case format_arg_v10:
            return FormatSelector::v10;
        default:
            throw std::invalid_argument(
		"Invalid DAQ format version specifier"
		);
    }
}

//____________________________________________________________________________
/**
 * @brief Make a data source using the source string and appropriate factory.
 *
 * @param pFactory Pointer to the appropriate ring item factory.
 * @param strUrl Source URI string from args.
 *
 * @throw std::invalid_argument If the source has an unsupported protocol.
 *
 * @return Pointer to the created data source.
 *
 * @details Parse the URI of the source and:
 * - Based on the parse, create the underlying source.
 * - Create the concrete instance of the DataSource given all that.
 * 
 * @note The factory's getItem() method is used to pass undifferentiated ring 
 * itmes to the part of the code that does the ROOT I/O.
 * @note (ASC 9/11/23): Only file data sources are supported. 
 */
static DataSource*
makeDataSource(RingItemFactoryBase* pFactory, const std::string& strUrl)
{
    URL uri(strUrl);
    std::string protocol = uri.getProto();

    if (strUrl == "-") {
	std::string msg = "This version of eeconverter was not built with " \
	    "standard input data source support. Please specify a file data " \
	    "source and run the program again.\n";
	throw std::invalid_argument(msg);
    }
    
    if ((protocol == "tcp") || (protocol == "ring")) {
	std::string msg = "This version of eeconverter was not built with " \
	    "ringbuffer data source support. Please specify a file data " \
	    "source and run the program again.\n";
	throw std::invalid_argument(msg);
    } else {
	std::string path = uri.getPath();
	// Need it to last past block:
        std::ifstream& in(*(new std::ifstream(path.c_str())));
	std::cout << "Input file path: " << path << std::endl;
	
        return new StreamDataSource(pFactory, in);
    }
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
  
    gengetopt_args_info args;
    cmdline_parser(argc, argv, &args);

    // Create the data source. Data sources allow us to specify ring item
    // types that will be skipped. They also allow us to specify types
    // that we may only want to sample (e.g. for online ring items).

    /**
     * @todo (ASC 9/8/23): If we ever want to convert from a ringbuffer or 
     * stream we need a method to make the source string if we choose 
     * some default i.e. "" --> tcp://localhost/username. For now, check the 
     * protocol, but:
     *   - This parameter is mandatory and
     *   - It has to be a file.
     */
    std::string sourceName(args.source_arg);
    std::string sinkName(args.fileout_arg);
    int skipCount = args.skip_given ? args.skip_arg : 0;
    if (skipCount < 0) {
	std::stringstream msg;
	msg << "--skip value must be >= 0 but is " << skipCount << std::endl;
	return EXIT_FAILURE;
    }    
    int itemCount = args.count_given ? args.count_arg : 0;
    if (itemCount < 0) {
	std::stringstream msg;
	msg << "--count value must be >= 0 but is " << itemCount << std::endl;
	return EXIT_FAILURE;
    }
    std::vector<uint32_t> exclusionList;
    try {
	exclusionList = makeExclusionList(args.exclude_arg);
    }
    catch (std::invalid_argument& e) {
	std::cerr << "Failed to create exclusion list: "
		  << e.what() << std::endl;
	return EXIT_FAILURE;
    }

    // Set the expected data format based on the enum value:
    FormatSelector::SupportedVersions version
    	= mapVersion(args.format_arg);
    auto& factory = FormatSelector::selectFactory(version);
    
    // Construct a data source using the factory and URI. Wrap it in a
    // unique_ptr so its automatically deleted when out of scope:
    DataSource* source;
    try {
    	source = makeDataSource(&factory, sourceName);
    } 
    catch (std::string msg) {
    	std::cerr << "Failed to open the data source " << sourceName << ": "
    		  << msg << std::endl;
    	return EXIT_FAILURE;
    }
    std::unique_ptr<DataSource> pSource(source);

    // Create the file sink:
    RootSinkProcessor processor(sinkName);
    
    // If there's a skip count, we can skip those items now:
    for (int i = 0; i < skipCount; i++) {
	std::unique_ptr<CRingItem> pItem(pSource->getItem());
	if (!pItem.get()) {
	    return EXIT_SUCCESS; // End of data source.
	}
    }
    
    // Dump items which are not excluded. If there's a dump count, dump that
    // many items, otherwise keep dumping until the end of the data source:
    int remain = itemCount;
    while (1) {
	std::unique_ptr<CRingItem> pItem(pSource->getItem());
	if (!pItem.get()) {
	    return EXIT_SUCCESS; // End of data source.
	}
	
	// If the item type is not in the exclusion list, its dumpable:
	if (std::find(
		exclusionList.begin(), exclusionList.end(), pItem->type()
		) == exclusionList.end()) {

	    try {
		processItem(*(pItem.get()), factory);
	    }
	    catch (std::logic_error& e) {
		std::cerr << "\nERROR: " << e.what() << std::endl;
		return EXIT_FAILURE;
	    }
	    
	    // itemCount >= 0 iff valid --count value is provided.
	    if (itemCount) {
		remain--;
		if (remain <= 0) {
		    return EXIT_SUCCESS;
		}
	    }
	}
    }
    
    // Fall through to here. Really should return before this.    
    return EXIT_SUCCESS;
}

//____________________________________________________________________________
/**
 * @details
 * Based on the item type, use the factory to get a new item using the same 
 * data for the appropriate type and pass it off to the processor.
 *
 * @note This method is rather long but this is only due to the switch 
 * statement that must handle every possible ring item type in DataFormat.h. 
 * The actual code is really quite simple (I think).
 * @note Use std::unique_ptr to ensure that temporary specific ring item 
 * objects are automatically deleted.
 * @todo (ASC 9/11/23): The data format item is only in the first run segment.
 * If provided another subrun, how do we check?
 */
void
CEventProcessor::processItem(const CRingItem& item, RingItemFactoryBase& factory)
{
    std::string dumpText;
 
    /**
     * @note The switch statement here assumes that if you have a ring item 
     * type the factory can generate it... this fails if the wrong version of 
     * the factory is used for the event file.
     */
    
    switch(item.type()) {
    case BEGIN_RUN:
    case END_RUN:
    case PAUSE_RUN:
    case RESUME_RUN:
    {
	std::unique_ptr<CRingStateChangeItem>
	    p(factory.makeStateChangeItem(item));
	dumpText = p->toString();
    }
    break;
    case ABNORMAL_ENDRUN:
    {
	std::unique_ptr<CAbnormalEndItem>
	    p(factory.makeAbnormalEndItem(item));
	dumpText = p->toString();
    }
    break;
    case PACKET_TYPES:
    case MONITORED_VARIABLES:
    {
	std::unique_ptr<CRingTextItem>
	    p(factory.makeTextItem(item));
	dumpText = p->toString();
    }
    break;
    case RING_FORMAT:
    {
	try {
	    std::unique_ptr<CDataFormatItem> p(factory.makeDataFormatItem(item));
	    dumpText = p->toString();
	}
	catch (std::bad_cast& e) {
	    std::string msg = "Unable to dump data format item. " \
		"Likely you've specified\n the wrong --format option.\n";
	    throw std::logic_error(msg);
	}
    }
    break;
    case PERIODIC_SCALERS:
        // case INCREMENTAL_SCALERS: // Same value as PERIODIC_SCALERS.
    case TIMESTAMPED_NONINCR_SCALERS:
    {
	std::unique_ptr<CRingScalerItem>
	    p(factory.makeScalerItem(item));
	dumpText = p->toString();
    }
    break;
    case PHYSICS_EVENT:
    {
	std::unique_ptr<CPhysicsEventItem>
	    p(factory.makePhysicsEventItem(item));
	dumpText = p->toString();
    }
    break;
    case PHYSICS_EVENT_COUNT:
    {
	std::unique_ptr<CRingPhysicsEventCountItem>
	    p(factory.makePhysicsEventCountItem(item));
	dumpText = p->toString();
    }
    break;
    case EVB_FRAGMENT:
    {
	std::unique_ptr<CRingFragmentItem>
	    p(factory.makeRingFragmentItem(item));
	dumpText = p->toString();
    }
    break;
    case EVB_UNKNOWN_PAYLOAD:
    {
	std::unique_ptr<CUnknownFragment> p(factory.makeUnknownFragment(item));
	dumpText = p->toString();
    }
    break;
    case EVB_GLOM_INFO:
    {
	std::unique_ptr<CGlomParameters>
	    p(factory.makeGlomParameters(item));
	dumpText = p->toString();
    }
    break;
    default:
    {
	std::unique_ptr<CUnknownFragment>
	    p(factory.makeUnknownFragment(item));
	dumpText = p->toString(); // Unknown item type.
    }
    break;
    }
 
    std::cout << "------------------------------------------\n";
    std::cout << dumpText << std::endl;
}
