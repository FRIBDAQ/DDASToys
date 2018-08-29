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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/**
 * Usage:
 *     binconvert infile outfile [numevents]
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
    s << "    asciiconvert infile outfile [numevents]\n";
    
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
    uint32_t crate = hit.GetCrateID();
    uint32_t slot  = hit.GetSlotID();
    uint32_t chan  = hit.GetChannelID();
    
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
        uint32_t tSize = trace.size();
        s.write(reinterpret_cast<const char*>(&tSize), sizeof(uint32_t));
        uint16_t traceArray[tSize];
        for (size_t i = 0; i < tSize; i++) {
            traceArray[i] =  trace[i];
        }
        s.write(
            reinterpret_cast<const char*>(traceArray),
            tSize * sizeof(uint16_t)
        );
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
    std::vector<DAQ::DDAS::DDASHit> result;
    if (item.type() == PHYSICS_EVENT) {
        FragmentIndex frags(reinterpret_cast<uint16_t*>(item.getBodyPointer()));
        FragmentInfo frag;
        DAQ::DDAS::DDASHitUnpacker unpacker;
        size_t nFrags = frags.getNumberFragments();
        for (size_t i =0; i < nFrags; i++) {
            FragmentInfo frag = frags.getFragment(i);
            DAQ::DDAS::DDASHit hit;
            uint32_t* begin = reinterpret_cast<uint32_t*>(frag.s_itembody);
            uint32_t* end   = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(frag.s_itembody) + frag.s_size
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
 * @param num - Number of hits to output
 */
static void
convert(CFileDataSource& in, std::ostream& out, uint64_t num)
{
    CRingItem* pItem;
    uint64_t nwritten = 0;
    while ((pItem = in.getItem())) {
        std::vector<DAQ::DDAS::DDASHit> hits = getHits(*pItem);
        for (size_t i = 0; i < hits.size(); i++) {
            if (writeMe(hits[i])) {
                writeHit(out, hits[i]);
                nwritten++;
                if (nwritten == num) {
                    delete pItem;
                    return;
                }
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
    uint64_t events = UINT64_MAX;
    if ((argc != 3) && (argc != 4))
        usage(std::cerr, "Invalid number of parameters");
    
    std::vector<uint16_t> empty;
    int fd;
    if (*argv[1] == '-') {
        fd = STDIN_FILENO;
    } else {
        fd = open(argv[1], O_RDONLY);        
    }
    
    if (fd < 0) {
        usage(std::cerr, "Input file could not be opened");
    }
    
    CFileDataSource src(fd, empty);
    std::ofstream   dest(argv[2]);
    
    if (argc == 4) {
        char* endptr;
        events = strtoull(argv[3], &endptr, 0);
        if (endptr == argv[3]) {
            usage(std::cerr, "Event count must be an unsigned integer");
        }
    }
    
    convert(src, dest, events);
    
    return EXIT_SUCCESS;
}