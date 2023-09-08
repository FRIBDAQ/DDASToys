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
 * @file  CEventProcessor.h
 * @brief Define an event proessor for reading ring items from a source. 
 * Generally a file for this application, creating an analysis pipleline to 
 * consume ring items from a ringbuffer is untested.
 */

#ifndef CEVENTPROCESSOR_H
#define CEVENTPROCESSOR_H

class URL;
class CRingItem;
class CRingItemProcessor;

/**
 * @class CEventProcessor
 * @brief Define an event processor for DDAS data.
 *
 * @details
 * The class defines a functor that can be created and invoked from main() to 
 * do the job of processing ring items from a source and handing them off to a 
 * data sink.
 */

class CEventProcessor
{
    // Private data:
  
private:
    URL* m_pDataSource; //!< Full data source URL
    CRingItemProcessor* m_pProcessor; //!< Processor to handle ring item types.
    unsigned m_itemCount; //!< Number of events to convert.
    unsigned m_skipCount; //!< Number of events to skip off the front.

    // Canonicals:
  
public:
    /** @brief Constructor. */    
    CEventProcessor();
    /** @brief Destructor. */
    ~CEventProcessor();
  
    // Entry point:
  
public:
    /**
     * @brief Entry point for the processor.
     * @param argc Number of command line arguments.
     * @param argv Command line arguments.
     * @return When done processing items.
     * @retval EXIT_SUCCESS Always.
     * @throw std::invalid_argument If gengetopts parameters are invalid.
     * @throw CException If the data source cannot be created.
     */
    int operator()(int argc, char* argv[]);

    // Utilities:
  
private:
    /**
     * @brief Type-independent processing of ring items. The processor will
     * handle specifically what to do with each ring item.
     * @param item  References the ring item we got.
     */    
    void processRingItem(CRingItem& item);
};

#endif
