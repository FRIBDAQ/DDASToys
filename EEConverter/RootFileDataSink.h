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
 * @file  RootFileDataSink.h
 * @brief Define a CDataSink that writes out result ring items to a ROOT file.
 */

#ifndef ROOTFILEDATASINK_H
#define ROOTFILEDATASINK_H

#include <CDataSink.h>

#include <vector>

class TTree;
class TFile;

class CRingItem;
namespace DAQ {
    namespace DDAS {
	class DDASFitHitUnpacker;
    }
}

class DDASRootFitEvent; // Holds the decoded event for output.
class RootHitExtension;

/**
 * @class RootFileDataSink
 * @brief This class knows how to write ROOT files from the ring items created
 * by the fitting program.
 * @note  Put is not intended to be used by this file. If it's used, a warning 
 * will be output to stderr. pData will then be treated as a raw ring item, 
 * turned into a CRingItem and putItem will be called from then on.
 */

class RootFileDataSink : public CDataSink
{  
public:
    /**
     * @brief Constructor.
     * @param filename  ROOT file to open. 
     * @param treename  Name of the tree to create in the root file. The tree 
     *   name defaults to "DDASFit" if not provided.
     * @throw ... All exceptions back to the caller.
     */
    RootFileDataSink(const char* filename, const char* treename="DDASFit");
    /** @brief Destructor. */
    virtual ~RootFileDataSink();
  
public:
    /**
     * @brief Put a ring item to file. 
     * @param item Reference to a ring item object.
     */
    virtual void putItem(const CRingItem& item);
    /**
     * @brief Called to put arbitrary data to the file. 
     * @param pData  Pointer to the data.
     * @param nBytes Number of bytes of data to put; actually ignored.
     */
    virtual void put(const void* pData, size_t nBytes);
  
private:
    DAQ::DDAS::DDASFitHitUnpacker* m_pUnpacker; //!< Unpacker for fit events.
    DDASRootFitEvent* m_pTreeEvent; //!< The ROOT-ized event to write.
    //std::vector<RootHitExtension> m_extensions; //!< Fit extensions.
    TTree* m_pTree; //!< Tree in the output file we write to.
    TFile* m_pFile; //!< The output ROOT file.
    bool m_warnedPutUsed; //!< Warning flag to call the right put.
};

#endif
