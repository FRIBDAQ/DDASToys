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

#include <string>
#include <vector>

class CRingItem;
class CRingItemProcessor;
class RingItemFactoryBase;

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
    // Canonicals:
  
public:
    /** 
     * @brief Constructor. 
     * @details
     * Nothing really happens here until operator() is called.
     */
    CEventProcessor() {};
    /** @brief Destructor. */
    ~CEventProcessor() {};

    /**
     * @brief Entry point for the processor.
     * @param argc Number of command line arguments.
     * @param argv Command line arguments.
     * @return When done processing items.
     * @retval EXIT_SUCCESS If data is processed successfully.
     * @retval EXIT_FAILURE Some part of the data processing pipeline cannot
     *   be created successfully.
     * @retval EXIT_FAILURE If the data format provided via the --format flag
     *   is incorrect.
     */
    int operator()(int argc, char* argv[]);

    // Utilities:
  
private:
    /**
     * @brief Type-independent processing of ring items. The processor will
     * handle specifically what to do with each ring item.
     * @param pItem Pointer to the ring item we got.
     * @param factory Reference to the factory appropriate to the format.
     * @throw std::logic_error Expected data format differs from what's read.
     */    
    //void processItem(CRingItem* pItem, RingItemFactoryBase& factory);
    void processItem(const CRingItem& pItem, RingItemFactoryBase& factory);
};

#endif
