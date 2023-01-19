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

/** @file:  v10to11converter.cpp
 *  @brief: Convert NSCLDAQ 10.x - 11.x event built data.
 */


/**
 * Usage:
 *     v10to11converter infile outfile
 *
 *  infile - an input event file
 *  outile - an output event file.
 *
 * The assumption is that the data are event built  nscldaq 10 has no body
 * headers.  Therefore the format of the input data  is:
 *
 * +--------------------------------
 * |    Ring item header           |
 * +-------------------------------+
 * |  Bytes in body (32 bits)      |
 * +-------------------------------+
 * |   frag 1 ...                  |
 * +-------------------------------+
 *     ...
 *
 *  Each fragment consists of the normal fragment header and a body that
 *  is a ring item that is in 10.x form.  We'll use the information in the
 *  body header to construct a body header for the output ring item and for
 *  each fragment ring item.
 *
 *  For simplicity we're only going to bother with physics items.
 *
 *  @note the nice thing a out ring items and their outer header is that
 *        11.x readers will read ring items just fine.. however we can't
 *        rely on those classes to figure out body headers and body pointers
 *        so we'll just use the resulting ring item item pointers from then on.
 *  @note FragmentIndex, other than getting the s_itembody wrong for these events
 *         will work perfectly well to bust up the eventbuilt data since it just
 *         cares about the fragment headers _except_ when computing s_itembody.
 */
#include <CFileDataSource.h>
#include <CFileDataSink.h>
#include <CRingItem.h>
#include <URL.h>
#include <FragmentIndex.h>
#include <Exception.h>
#include <DataFormat.h>
#include <fragment.h>

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>
#include <iostream>

/**
 * usage
 *    The usual error message + program usage to the output stream
 *    and exit.
 *    
 *  @param o   - the output stream to which the messages go.
 *  @param msg - the error message at program level.
 */
static void
usage(std::ostream& o, const char* msg)
{
    o << msg << std::endl;
    o << "Usage:\n";
    o << "   v10to11converter infile outfile\n";
    o << "Where: \n";
    o << "    infile   - is the name of the input file\n";
    o << "    outfile  - is the name of the outputf file\n";
    o << "NOTE: \n";
    o << "   infile outfile are file name paths not URIs.\n";
    
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
 * sizeOutputRingItem
 *    Determines the size of the final translated ring item.  This is
 *    computed as follows.  The output ring item has a ring item header,
 *    and a body header. The body has a uint32_t containing the bytes
 *    in the event body.
 *    Each fragment contains a fragment header and the payload with a body header
 *    added.
 *
 * @param frags  Reference to the FragmentIndex that describes the fragments.
 * @return uint32_t - the number of bytes in the translated ring item described by
 *         frags.
 */
static uint32_t
sizeOutputRingItem(FragmentIndex& frags)
{
    uint32_t result = sizeof(RingItemHeader) + sizeof(BodyHeader); // header + bodyhdr 
    result += sizeof(uint32_t);            // Size of fragment data.
    
    for (auto p = frags.begin(); p != frags.end(); p++) {
        result += sizeof(EVB::FragmentHeader);  // Each frag has a header.
        result += sizeof(BodyHeader) + p->s_size; // Size of translated fragment.
    }
    return result;
}
/**
 * insertTranslatedFragment
 *    Translate a fragment and insert it into the ring item at its cursor.
 *
 *   @param item  - Reference to the ring item we're creating.
 *   @param frag  - Fragment info of the fragment.
 *
 */
static void
insertTranslatedFragment(CRingItem& item, FragmentInfo& frag)
{
    // What we're going to put is a fragment header and a ring item.
    
    uint8_t* pdest = reinterpret_cast<uint8_t*>(item.getBodyCursor()); 
    
    
    // Build and insert the fragment header from the frag:
    
    EVB::FragmentHeader fragHeader;
    fragHeader.s_timestamp = frag.s_timestamp;
    fragHeader.s_sourceId  = frag.s_sourceId;
    fragHeader.s_size      = frag.s_size + sizeof(BodyHeader); // frags get body headers.
    fragHeader.s_barrier   = frag.s_barrier;
    
    memcpy(pdest, &fragHeader, sizeof(EVB::FragmentHeader));
    pdest += sizeof(EVB::FragmentHeader);  // Ring item goes here.
    
    // Construct a ring item consisting of the payload and the appropriate body header.
    
    CRingItem payload(
        PHYSICS_EVENT, frag.s_timestamp, frag.s_sourceId, frag.s_barrier,
        fragHeader.s_size
    );
    // Fill in the bodyCursor:
    
    uint8_t *pPayload  =
        reinterpret_cast<uint8_t*>(payload.getBodyCursor());  // After body header.
    
    // We need to strip off the ring item header from the stuff we're copying in.
    
    size_t nToCopy = frag.s_size - sizeof(RingItemHeader);
    uint8_t* pSrc  = reinterpret_cast<uint8_t*>(frag.s_itemhdr);
    pSrc          += sizeof(RingItemHeader);
        
    memcpy(pPayload, pSrc, nToCopy);
    pPayload += nToCopy;
                                    // Ring item size book keeping.
    payload.setBodyCursor(pPayload);
    payload.updateSize();
    
    // Now copy this newformed ring item into the output ring item and update the
    // cursor (caller is responsible for doing an updateSize when all is done).
    
    pRingItemHeader pHeader =
        reinterpret_cast<pRingItemHeader>(payload.getItemPointer());
    memcpy(pdest, pHeader, pHeader->s_size);
    pdest += pHeader->s_size;
    
    item.setBodyCursor(pdest);
}
/**
 * translateItem
 *    Translate a 10.x ring item of event built data into 11.x event buildt data
 *    There's enough information to allow us to construct the proper body
 *    headers etc.
 *
 *  @param pIn    - Input ring item pointer.
 *  @return CRingItem* - Dynamically created output ring item (must be deleted by caller).
 */
CRingItem*
translateItem(CRingItem* pIn)
{
    // Figure out the body pointer for 10.x items:
    // Note that ring item headers are the same for 10 and 11 so this is ok:
    
    void* pItem = pIn->getItemPointer();     // Points to the ring item.
    pRingItemHeader pHeader = reinterpret_cast<pRingItemHeader>(pItem);
    
    // The body is just past the header - unconditionally for 10:
    // The body should be (mostly) decodable into fragments by FragmentIndex
    // The s_itembody pointer, is, however not reliable, though s_itemhdr is
    // and points to the ring item header of each item.
    // There must always be at least one fragment (I think).    
    uint16_t* pBody = reinterpret_cast<uint16_t*>(pHeader + 1);
    FragmentIndex frags = FragmentIndex(pBody);
    if (frags.getNumberFragments() == 0) {
        throw std::string("Encountered an input item with no fragments");
    }
    
    // Figure out the size of the output ring item:

    size_t totalSize = sizeOutputRingItem(frags);
    size_t bodySize  = totalSize - sizeof(RingItemHeader) -sizeof(BodyHeader);   // Size of body.
    
    // Create the ring item of the appropriate size and insert the
    // body size:
    
    FragmentInfo first = frags.getFragment(0);
    CRingItem* pResult = new CRingItem(
        PHYSICS_EVENT, first.s_timestamp, first.s_sourceId, first.s_barrier,
        totalSize
    );
    uint32_t *pSize = reinterpret_cast<uint32_t*>(pResult->getBodyCursor());
    *pSize++ = bodySize;
    pResult->setBodyCursor(pSize);
    
    // Now insert each translated fragment:
    
    for(auto p = frags.begin(); p != frags.end(); p++) {
        insertTranslatedFragment(*pResult, *p);
    }
    
    pResult->updateSize();        // Get the header, body header all straight.
    return pResult;
}

/**
 * main
 *    Entry point:
 *       - Verify the right number of parameters.
 *       - Construct the data source and sink.
 *       - convert the file.
 *       
 * @param argc - number of command line words.
 * @param argv - array of pointers to the command line words.
 * @return int - hopefully EXIT_SUCCESS.
 */
int
main(int argc, char** argv)
{
    // Check for the right number of parameters:
    
    if (argc != 3) {
        usage(std::cerr, "Incorrect number of command parameters");
    }
    
    const char* infile = argv[1];
    const char* outfile = argv[2];
    
    // Try to open the input file:
    
    CFileDataSource* pSource(0);
    try {
        URL uri = filenameToUri(infile);   // This wants an URI for input.
        std::vector<uint16_t> excludes;
        pSource = new CFileDataSource(uri, excludes);
        
    }
    catch (CException& e) {
        std::cerr << "Input file open failed: " << e.ReasonText() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (std::exception e) {
        std::cerr << "Input file open failed: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Input file open failed: " << "Unexpected exeption type" << std::endl;
        throw;                // So we get a handle on the exception type and can add a handler.
    }
    CFileDataSource& source(*pSource);
    
    // Try to open the output file:
    
    CFileDataSink* pSink(0);
    try {
        pSink = new CFileDataSink(outfile);
    }
    catch (CException& e) {
        std::cerr << "Output file open failed: " << e.ReasonText() << std::endl;
        delete pSource;
        exit(EXIT_FAILURE);
    }
    catch (std::exception e) {
        std::cerr << "Output file open failed: " << e.what() << std::endl;
        delete pSource;
        exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Output file open failed: " << "Unexpected exeption type" << std::endl;
        delete pSource;
        throw;                // So we get a handle on the exception type and can add a handler.
    }
    
    CFileDataSink& sink(*pSink);
    
    // Loop over the data:
    
    CRingItem* pItem(0);
    try {
        
        while(pItem = source.getItem()) {
            if (pItem->type() == PHYSICS_EVENT) {  // Only output physics items.
                CRingItem* pOutputItem = translateItem(pItem);
                sink.putItem(*pOutputItem);
                delete pOutputItem;
            }
            delete pItem;
            pItem = nullptr;           // So catch handlers don't fail on deletes.
        }
    }
    catch (CException& e) {
        std::cerr << "Ring item processing failed: " << e.ReasonText() << std::endl;
        delete pSource;
        delete pItem;
        exit(EXIT_FAILURE);
    }
    catch (std::exception e) {
        std::cerr << "Ring item processing failed: " << e.what() << std::endl;
        delete pSource;
        delete pItem;
        exit(EXIT_FAILURE);
    }
    catch (std::string msg) {
        std::cerr << "Ring item processing failed: " << msg << std::endl;
        delete pSource;
        delete pItem;
        exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Ring item processing failed: " << "Unexpected exeption type" << std::endl;
        delete pSource;
        delete pItem;
        throw;                // So we get a handle on the exception type and can add a handler.
    }
    // We got here so everything worked:
    
    delete pSource;
    sink.flush();
    delete pSink;
    
    exit(EXIT_SUCCESS);
}