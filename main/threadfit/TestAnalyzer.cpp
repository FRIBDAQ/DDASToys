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

/** @file:  TestAnalyzer.cpp
 *  @brief: Provides a testing harness for CDDASAnalyzer.
 */

/**
 * This program reads data from event files using the old fashioned tried and
 * true CFileDataSource to get a ring item at a time.  Ring items are then
 * sent to an instance of CDDASAnalyzer which is equipped with an outputter that
 * just writes the data back to disk along with the fitted traces, if appropriate.
 */

/**
 * Usage:
 *    TestAnalyzer input-file output-file
 *
 *    Where:
 *        input-file - is the path to the input event file (we'll turn that into
 *                     a URI).
 *        output-file - The path to the file we're going to write.
 */


#include <CFileDataSource.h>
#include <CFileDataSink.h>
#include <CRingItem.h>
#include <Exception.h>
#include <DataFormat.h>
#include <FragmentIndex.h>
#include <fragment.h>
#include <DDASHitUnpacker.h>

#include <stdexcept>
#include <string>
#include "CDDASAnalyzer.h"
#include <URL.h>
#include "Outputter.h"
#include "FitHitUnpacker.h"
#include "DDASFitHit.h"

#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static const int id(1);              // Fake thread/source id.


/**
 *   My fake outputter... reconstructs ring items from the
 *   fragment list and then submits those to the data sink for output.
 *   For now end does nothing.  In production it probably should destroy the sink
 *   and make future outputItem calls fail miserably.
 */

class CFileOutputter : public Outputter
{
private:
    CFileDataSink  m_sink;
public:
    CFileOutputter(const char* filename);
    virtual void outputItem(int id, void* pItem);
    virtual void end(int id);
private:
    void deleteIfNeeded(FragmentInfo& frag);
};

// Controls the fits.

class MyPredicate : public CDDASAnalyzer::FitPredicate
{
public:
    ~MyPredicate() {}
    std::pair<std::pair<unsigned, unsigned>, unsigned> operator() (
        const FragmentInfo& frag, DAQ::DDAS::DDASHit& hit,
        const std::vector<uint16_t>& trace
    );
};

// Only fit crate 0, slot2, channel 0

std::pair<std::pair<unsigned, unsigned>, unsigned>
MyPredicate::operator() (
        const FragmentInfo& frag, DAQ::DDAS::DDASHit& hit,
        const std::vector<uint16_t>& trace
)
{
    if (
        (hit.GetCrateID() == 0) && (hit.GetSlotID() == 2) &&
        (hit.GetChannelID() == 0)
    ) {
        return
            std::pair<std::pair<unsigned, unsigned>, unsigned>(
                (0, trace.size() - 1), 0xffff
            );
    
    } else {
        return std::pair<std::pair<unsigned, unsigned>, unsigned>(
            (0, 0), 0xffff
        );   // supresses the fit.
    }
}

MyPredicate pred;


/**
 * usage
 *   Output message with usage and exit.
 * @param o - output stream.
 * @param msg - error message.
 * @note - does not return.
 */
static void
usage(std::ostream& o, const char* msg)
{
    o << msg << std::endl;
    o << "Usage:\n";
    o << "    TestAnalyzer input-file output-file\n";
    o << "Where\n";
    o << "    input-file is the path to the file to analyze\n";
    o << "    output-file is the path to the output file\n";
    
    exit(EXIT_FAILURE);
    
}
/**
 * filenameToUri
 *    Constructs the file URI that's assocaiated with the filename.
 *
 * @param name - name of the file to URI-ize.
 * @return URL - object wrapping the URI.
 */
static URL
filenameToUri(const char* name)
{
    char* absoluteName = canonicalize_file_name(name);
    std::string uriString = "file://";
    uriString += absoluteName;
    URL result(uriString);
    free(absoluteName);
    
    return result;
}


/**
 * main
 *    Entry point for the program.
 *
 * @param argc - count of command line parameters.
 * @param argv - The parameters.
 * @return int - hopefully EXIT_SUCCESS
*/
int
main(int argc, char** argv)
{
    if (argc != 3) {
        usage(std::cerr, "Incorrect number of command line parameters");
    }
    
    const char* infile = argv[1];
    const char* outfile = argv[2];
    
    // We'll lump all file open failures in a single try /catch handler.
    
    
    CFileDataSource* pSource(0);
    CFileOutputter* pOutputter(0);
    try {
        URL eventFile = filenameToUri(infile);
        std::vector<uint16_t> empty;
        pSource = new CFileDataSource(eventFile, empty);
        
        pOutputter = new CFileOutputter(outfile);
    }
    catch (CException& e) {
        delete pSource;
        delete pOutputter;
        usage(std::cerr, e.ReasonText());
    }
    catch (std::exception& e) {
        delete pSource;
        delete pOutputter;
        usage(std::cerr, e.what());
    }
    catch (const char* msg) {
        delete pSource;
        delete pOutputter;
        usage(std::cerr, msg);
    }
    catch (std::string msg ) {
        delete pSource;
        delete pOutputter;
        usage(std::cerr, msg.c_str());
    }
    catch (...) {
        usage(std::cerr, "Some error occured opening the input or output file");
    }
    /**
     *  Construct the analyzer... I don't think that can throw so:
     */
    
    CDDASAnalyzer analyzer(id, *pOutputter);
    CRingItem* pItem;
    analyzer.setSingleFitPredicate(&pred);
    analyzer.setDoubleFitPredicate(&pred);
    // Process the data:
    
    try {
        while(pItem = pSource->getItem()) {
            analyzer(pItem);
            delete pItem;
        }
    }
    catch (CException& e) {
        std::cerr << "Failure processing ring item: " << e.ReasonText() << std::endl;
        delete pSource;
        delete pOutputter;
        exit(EXIT_FAILURE);
    }
    catch (std::exception& e) {
        std::cerr << "Failure processing ring item " << e.what() << std::endl;
        delete pSource;
        delete pOutputter;
        exit(EXIT_FAILURE);
    }
    catch (const char* msg) {
        std::cerr << "Failure processing ring item: " << msg << std::endl;
        delete pSource;
        delete pOutputter;
        exit(EXIT_FAILURE);
    }
    catch (std::string msg) {
        std::cerr << "Failure processing ring item: " << msg << std::endl;
        delete pSource;
        delete pOutputter;
        exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Failure processing ring item; generalized exception handler\n";
        delete pSource;
        delete pOutputter;
        exit(EXIT_FAILURE);
    }
    
    // Destroy/shutdown the source and outputter.
    
    delete pSource;
    delete pOutputter;
}
/*--------------------------------------------------------------------------
 *  Implement methods in CFileOutputter -- who  knows we may pull this out
 *  and use it in the application.
 */

/**
 * constructor
 *   @param filename
 */


CFileOutputter::CFileOutputter(const char* filename) :
    m_sink(std::string(filename))
{   
}

/**
 * end
 *    Just give the sink a flush
 *
 *  @param id - node id ignored.
 */
void
CFileOutputter::end(int id) {
    m_sink.flush();
}

/**
 * outputItem
 *     Outputs a result item.
 *
 *  @param id -- thread id (we plug this in as the data source in the body header).
 *  @param pItem - Actually a  pointer to CDDASAnalyzer::outData
 *                 the payload part of this is a pointer to an std::vector<FragmentInfo>
 *                 Each fragment is either an unmodified DDAS hit without a trace or
 *                 a DDAS hit with a trace to which a HitExtension has been appended.
 *  @note Hits that have traces and a hit extension must be free'd as they were
 *        originally malloced.
 *
 *
 *  What we write to file is a ring item that looks like it came from the
 *  event builder.  id is our sourceid, the timestamp comes from pItem,
 *  The body has the usual total byte count and then the fragments.
 */
void
CFileOutputter::outputItem(int id, void* pData)
{
    // First the pData looks like a DDASAnalyzer::outData struct:
    
    CDDASAnalyzer::outData* pRawEvent =
        reinterpret_cast<CDDASAnalyzer::outData*>(pData);
    
    // The payload is a pointer to a vector of FragmentInfo.  Note that
    // frag.s_itemhdr and frag.s_itembody will, if there's a waveform,
    // be dynamic and frag.s_itemhdr will have to be free(3)ed when we're done
    // with it.
    
    std::vector<FragmentInfo>& fragments(
        *reinterpret_cast<std::vector<FragmentInfo>* >(pRawEvent->payload)
    );
    // Size the fragment bodies.  From that we figure out the size
    // of the ring item....with slop because I'm cautious.
    // This initial size represents a ring item with a body header and that
    // total size longword.
    //
    size_t totalSize =
        sizeof(uint32_t) + sizeof(RingItemHeader) + sizeof(BodyHeader);
    uint32_t bodySize(sizeof(uint32_t));
    for (int i = 0; i < fragments.size(); i++) {
        uint32_t itemSize = fragments[i].s_size;
        totalSize += itemSize;;           // I think this is all inclusive.
        totalSize += sizeof(EVB::FragmentHeader);
        bodySize += itemSize  + sizeof(EVB::FragmentHeader);
    }
    
    totalSize += 100;                             // Slop.
    
    CRingItem outputItem(PHYSICS_EVENT, pRawEvent->timestamp, id, 0, totalSize);
    void* p = outputItem.getBodyCursor();   // Allows for body header...
    
    // Put in the total item size .. this is bodySize
    
    memcpy(p, &bodySize, sizeof(uint32_t));
    p = reinterpret_cast<void*>(reinterpret_cast<uint32_t*>(p) + 1);
    
    // put each fragment into the ring. I _think_ s_size of the fragment info
    // includes the fragment header and ring item and all that shit.
    // Fragments that have waveforms, need to have their storage freed as they
    // were dynamically allocated.  We'll do that if the fragment throws an
    // exception when parsed by the DDASHitUnpacker (since it'll think the length)
    // is inconsistent.
    //
    for (int i =0; i < fragments.size(); i++) {
        // Fragment header
        
        EVB::FragmentHeader h;
        h.s_timestamp = fragments[i].s_timestamp;
        h.s_sourceId = fragments[i].s_sourceId;
        h.s_barrier  = fragments[i].s_barrier;
        h.s_size     = fragments[i].s_size;
        
        memcpy(p, &h, sizeof(EVB::FragmentHeader));
        p  = reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(p) + sizeof(EVB::FragmentHeader)
        );
        
        // Fragment ring item:
        memcpy(p, fragments[i].s_itemhdr, fragments[i].s_size); // I think that's right.
        p = reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(p) + fragments[i].s_size
        );
        deleteIfNeeded(fragments[i]);
    }
    // At this point the ring item just needs to have its pointers updated
    // which updates the header too:
    
    
    outputItem.setBodyCursor(p);     
    outputItem.updateSize();         // Get the size right
    m_sink.putItem(outputItem);      // Write the item to file
}
/**
 * deleteIfNeeded
 *    If the ring item contained a fit we need to free it as it was dynamically
 *    allocated.  This is done in a bit of a dirty way;   If we added an
 *    extension to this, the the DDASHitUnpacker will think this item has
 *    a bad size and throw an std::runtime_error -- we'll catch that and
 *    free the dynamic storage in the cathc handler.
 * @param item - FragmentInfo for the item.
 */
void
CFileOutputter::deleteIfNeeded(FragmentInfo& info)
{
   
    DAQ::DDAS::FitHitUnpacker unpacker;
    DAQ::DDAS::DDASFitHit     hit;
    // Figure out the body start and end+1 pointers:
    
    uint32_t* pStart = reinterpret_cast<uint32_t*>(info.s_itemhdr);

    try {
        unpacker.decode(pStart, hit);
        if(hit.hasExtension()) free(info.s_itemhdr);
    }
    catch (...) {
        std::cerr << "Fileoutputter caught exception\n";
    }
}