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
 * @file DDASDecoder.h
 * @brief Define an event proessor for reading ring items from a source, in 
 * this case a file. Very similar to the CEventProcessor class but really 
 * designed to work interactively with the GUI.
 */

#ifndef DDASDECODER_H
#define DDASDECODER_H

#include <string>
#include <vector>

class URL;
class CRingItem;
class CDataSource;

namespace ddastoys {    
    class DDASFitHit;
}

class TraceViewProcessor;

/**
 * @class DDASDecoder
 * @brief An interactive event processor integrated in the traceview GUI.
 *
 * @details
 * An event processor class broken into parts performing specific actions 
 * like creating a data source or getting the next PHYSICS_EVENT. These 
 * functions can be hooked into the signal and slot framework used by Qt. 
 * See latest $DAQROOT/share/recipes/process/process.cpp for a more general 
 * example.
 */

class DDASDecoder
{
public:
    /** @brief Constructor. */
    DDASDecoder();
    /** @brief Destructor. */
    ~DDASDecoder();

    /**
     * @brief Create a file data source from the input string.
     * @param src  Name of the file we will create a data source from.
     */
    void createDataSource(std::string src);
    /**
     * @brief Get the next unpacked PHYSICS_EVENT.
     * @return The event data. The vector is empty if the end of the data 
     *   file is encountered.
     */
    std::vector<ddastoys::DDASFitHit> getEvent();
    /**
     * @brief Skip events in the currently loaded data file.
     * @param nevts Number of PHYSICS_EVENTS to skip.
     * @return int
     * @retval  0 END_RUN state change event is not encountered when 
     *   skipping.
     * @retval -1 END_RUN state change event is encountered when skipping.
     */
    int skip(int nevts);
    /**
     * @brief Return the number of PHYSCIS_EVENTs.
     * @return Number of PHYSICS_EVENTs in the file.
     */
    int getEventCount() { return m_count; };
    /**
     * @brief Return the PHYSICS_EVENT index.
     * @return Index of the event. Trivially m_count-1.
     */
    int getEventIndex() { return m_count-1; };
    /**
     * @brief Return the path of the file data source.
     * @return The file path. Returns an empty string if the data source 
     *   has not been created.
     */
    std::string getFilePath();
      
private:
    URL* m_pSourceURL;      //!< URL-formatted data source name.
    CDataSource* m_pSource; //!< Data source to read from.
    TraceViewProcessor* m_pProcessor; //!< TraceView processor for DDAS.
    int m_count; //!< How many PHYSICS_EVENTs have been processed.
    
    /**
     * @brief Get the next PHYSICS_EVENT ring item.
     * @return Pointer to the ring item.
     * @retval nullptr If there are no more PHYSICS_EVENTs in the file.
     */
    CRingItem* getNextPhysicsEvent();
    /**
     * @brief Perform type-independent processing of ring items. 
     * @param item References the ring item we got.
     */
    void processRingItem(CRingItem& item);  
};


#endif
