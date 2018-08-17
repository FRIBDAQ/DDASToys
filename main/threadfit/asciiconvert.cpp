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

/** @file:  asciiconvert.cpp
 *  @brief: Selectively convert traces in a ring item file to ASCII.
 */

#include <CFileDataSource.h>
#include <DDASHit.h>
#include <CRingItem.h>
#include <FragmentIndex.h>
#include <DataFormat.h>
#include <DDASHitUnpacker.h>



#include <iostream>
#include <fstream>
#include <vector>

#include <stdlib.h>
#include <stdint.h>


/**
 * Usage:
 *     asciiconvert infile outfile
 *
 *  infile - the input ring item file.
 *  outfile - the output ASCII file.
 *
 * Note this program needs to be edited :-(  to provide the predicate
 * that will be used to select traces from the file. Specifically the
 * writeMe function has to be written to say yeah or nay for a trace.
 *
 * Format of the output file is a sequence of traces.  Each trace has a one
 * line header that consists of the number of samples in the trace.
 * This is followed by that number of lines, each containing a sequential
 * trace sample.
 */

/**
 * usage
 *    Output an error message and program usage.
 *
 * @param s -output stream.
 * @param m -error message.
 * @note - exits with EXIT_FAILURE
*/
static void
usage(std::ostream& s, const char* msg)
{
    s << msg << std::endl;
    s << "Usage\n";
    s << "    asciiconvert infile outfile\n";
    
    exit(EXIT_FAILURE);
}

/**
 * writeMe
 *    Predicate like function that filters out the hits in the events
 *    that need to have their traces written.
 *
 * @param hit   - reference to a hit.
 * @return bool - true if the hit makes the criteria to write it out.
 */
bool writeMe(const DAQ::DDAS::DDASHit& hit) {
    uint32_t crate = hit.GetCrateId();
    uint32_t slot  = hit.GetSlotId();
    uint32_t chan  = hit.GetChannelId();
    
    return (crate == 0) && (slot == 2) && (chan == 0);  // Dynode channel.
}

/**
 * writeHit
 *   Write the trace (if there is one) from a hit.  If there is no trace
 *   it is not written silently.
 *
 * @param s - stream to which the data are written.
 * @param hit - Reference to the hit to write.
 */
static void
writeHit(std::ostream& s, const DAQ::DDAS::DDASHit& hit)
{
    if (hit.GetTraceLength() > 0) {
        const std::vector<uint16_t>& trace(hit.GetTrace());
        s << trace.size() << std::endl;
        for (int i = 0; i < trace.size(); i++) {
            s << trace[i] << std::endl;
        }
    }
}
/**
 * getHits
 *    Turns an event into a possibly empty set of DDAS hits.
 *    - If an event is not a PHYSICS_EVENT just return an empty
 *      vector.
 *    - Otherwise use fragmentindex and the fact that each hit is bundled into
 *      its own fragment to create the DDASHit vector which is returned.
 *
 * @param item -reference to the ring item to analyze.
 * @return std::vector<DAQ::DDAS::DDASHit>
 */
static std::vector<DAQ::DDAS::DDASHit>
getHits(CRingItem& item)
{
    std::vector<DAQ::DDAS::DDASHit> result
    if (item.type() == PHYSICS_EVENT) {
        FragmentIndex frags(reinterpret_cast<uint16_t*>(pItem->getBodyPointer()));
        FragmentInfo frag;
        DDASHitUnpacker unpacker;
        sizse_t nFrags = frags.getNumberFragments();
        for (int i =0; i < nFrags; i++) {
            FragmentInfo frag = frags.getFragment(i);
            DDASHit hit;
            uint32_t* begin = reinterpret_cast<uint32_t*>(f.s_itembody);
            uint32_t* end   = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(f.s_itembody) + f.s_size
            );
            unpacker.unpack(begin, end, hit);
            result.push_back(hit);
        }
    }
    
    return result;
}
/**
 * convert
 *   Does the actual convert.  Ring items are gotten from the file,
 *   ddas hits are extracted and, for each hit that statisfies the
 *   writeMe criteria, tha hit's trace is output.
 *
 * @param in - input data source.
 * @param out - output stream
 */
stati void
convert(CFileDataSource& in, std::ostream& out)
{
    CRingItem* pItem;
    while (pItem = in.getItem()) {
        std::vector<DAQ::DDAS::DDASHit> hits = getHits(*pItem);
        for (int i = 0; i < hits.size(); i++) {
            if (writeMe(hits[i])) {
                writeHit(out, hits[i]);
            }
        }
        delete pItem;
    }
}

/**
 * main
 *    Entry point
 *    Just do parameter parsing and set up the input/output objects before
 *    starting the conversion.
 */
int main(int argc, char** argv)
{
    if (argc != 3) usage(std::cerr, "Invalid number of parameters");
    
    std::vector<uint16_t> empty;
    CFileDataSource src(argv[1], empty);
    std::ofstream   dest(argv[2]);
    
    convert(src, dest);
    
    return EXIT_SUCCESS;
}