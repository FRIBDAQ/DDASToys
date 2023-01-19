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

/** @file:  CRingThreadedBlockReader.cpp
 *  @brief: Implement the threade block mode ring item reader.
 */

#include "CRingThreadedBlockReader.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include "zhelpers.hpp"
#include <stdexcept>
#include <system_error>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sstream>

/*----------------------------------------------------------------------
*  Implement the outer class. This class runs in the parent thread
*  with the exception of the static method readerEntry which is the entry
*  point of the reader thread.  See the implementation of the
*  nested, Reader class for information about it as it's implemented in that
*  section of this file.
*/


/**
 * constructor (CRingThreadedBlockReader)
 *
 *  Create the user interface object for the client thread, set up the
 *  parent side sockets and start the reader thread going.
 *
 *  @param file    - Path to the file to open.
 *  @param ctx     - References a zmq::context_t that will be used to create
 *                   the inproc sockets.
 *  @param bytes   - Size of the blocks the reader wil read.
 *  @param credits - Number of in flight read  messages allowed between
 *                   reader and receiver.
 *  @param creditWaterMark -  How many credits must accumulate before we
 *                   send new credits to the reader.  Note that this must be
 *                   less than credits.
 */
CRingThreadedBlockReader::CRingThreadedBlockReader(
    const char* file, zmq::context_t& ctx, size_t bytes,
    size_t credits, size_t creditWaterMark
) : m_readerThreadId(0), m_zmqContext(ctx), m_credit(ctx, ZMQ_PUSH),
    m_data(ctx, ZMQ_PULL), m_nBlocksize(bytes), m_nCredits(credits),
    m_nCreditsRemaining(credits), m_nCreditWaterMark(creditWaterMark),
    m_nFd(-1), m_partialItemSize(0), m_partialItemBlockSize(0),
    m_pPartialItem(nullptr), m_fEOF(false)
{
    // Open the file:
    
    m_nFd = open(file, O_RDONLY);
    if (m_nFd < 0) {
    throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)),
			    "Opening the ring item file");
    }
    
    // Bind the sockets (these are both servers).
    
    setupSockets();
    
    // Start the reader.
    
    startReader();
    
    // Send our initial credit message.
    
    sendCredits(m_nCreditsRemaining);
    
    
    // From now on our actions (reads on data socket and pushes of more credits)
    // are driven by the client calling read().
    
}

/**
 * destructor(CRingThreadedBlockReader).
 *   This must only be called when the entire file has been read.
 *   It assumes the reader has been joined with and that the
 *   sockets have been shutdown by the read() that saw the eof message
 *   and therefore also sent the done message to the Reader.
 */
CRingThreadedBlockReader::~CRingThreadedBlockReader()
{
    delete []m_pPartialItem;         // Free dynamic sstorage.
}

/**
 * read
 *    Gets the next message from the data socket using readData.
 *
 * @return DataDescriptor*  - Pointer to a dynamically allocated data
 *                            descriptor.  A size of 0 in s_nBytes and/or
 *                            s_nItems indicates the file was completely consumed.
 */
CRingThreadedBlockReader::DataDescriptor
CRingThreadedBlockReader::CRingThreadedBlockReader::read()
{
    DataDescriptor result = readData();
    if (result.s_nBytes == 0) {
        // readData got an EOF message:
        
        shutdown();               // Send 'done'.
        m_fEOF = true;            // Indicate we're not getting any more data.
    } else {                      // Not an EOF .. may  need to send more credits.
        m_nCreditsRemaining++;    // Count that we have data
        if (m_nCreditsRemaining > m_nCreditWaterMark) {
            sendCredits(m_nCreditsRemaining);
        }
        processDataMessage(result);   // figure out m_nItems, save any partial item.
    }
    return result;   
}
/**
 * creditServiceName
 *    Compute the service name for the credit socket.  In order to support
 *    multiple input streams that will be of the form:
 *      credit:fd  where fd is the file descriptor number.
 */
std::string
CRingThreadedBlockReader::creditServiceName() const
{
    return serviceName("credit");
}
/**
 * dataServiceName
 *    See above, but the servic name is for the data socket.
 */
std::string
CRingThreadedBlockReader::dataServiceName() const
{
    return serviceName("data");
}
/*---------------------------------------------------------------------------
 * Private methods for CRingThreadedBlockReader -- the outer class
 */

/**
 * Save the partial item in the m_pPartialItem block.  If necessary
 * That's resized to fit.
 *   @param pItem - pointer to the partial item.
 *   @param nBytes - number of bytes of partial item.
 *
 *  @note The dance we do here is intended to ensure we only 
 *        sometimes need to allocated storage for the partial.
 */
void
CRingThreadedBlockReader::savePartialItem(void* pItem, size_t nBytes)
{
  if (m_partialItemBlockSize < nBytes) {
    delete []m_pPartialItem;  	// No-op for null pointer.
    m_pPartialItem = new char[nBytes];
    m_partialItemBlockSize = nBytes;
  }
  memcpy(m_pPartialItem, pItem, nBytes);
  m_partialItemSize = nBytes;
  
}

/**
 * sendCredits
 *     Send a credit message to the child.  m_nCreditsRemaining is
 *     decreased by the number of credits sent.
 *
 * @param num - number of credits to send.
 */
void
CRingThreadedBlockReader::sendCredits(int num)
{
    s_sendmore(m_credit, "credit");
    zmq::message_t credits(sizeof(int));
    memcpy(credits.data(), &num, sizeof(int));
    
    m_credit.send(credits);
    m_nCreditsRemaining -= num;
}
/**
 * readData
 *    Reads a message from the data socket.  There are two cases:
 *    *  The message is a 'data' message - in which case we pass the
 *       second message part back to the caller.
 *    *  The message is an 'eof' message - in which case we create a
 *       fake descriptor with s_nByrtes and s_nItems zeroed out indicating
 *       and end file.
 *
 *  @return DataDescriptor - data descriptor.  Note that since we share
 *           address space with the reader, the pointer to the data is valid
 *           in our thread context too.
 *        
 */
CRingThreadedBlockReader::DataDescriptor
CRingThreadedBlockReader::readData()
{
    std::string type = s_recv(m_data);
    DataDescriptor result;
    
    if (type == "data") {
        zmq::message_t dataitem;
        m_data.recv(&dataitem);
        assert(dataitem.size() == sizeof(DataDescriptor));  // Must be a data desc.
        memcpy(&result, dataitem.data(), sizeof(DataDescriptor));
    } else if (type == "eof") {
        result.s_nBytes = 0;
        result.s_nItems = 0;
        result.s_pData  = nullptr;          // Ensure dereference crashes.
    } else {
        throw std::logic_error("CRingThreadedBlockReader::s_recv - got a bad message type");
    }
    return result;
}
/**
 * setupSockets
 *    Sets up the m_credit and m_data sockets as servers.
 */
void
CRingThreadedBlockReader::setupSockets()
{
    m_credit.bind(creditServiceName().c_str());
    m_data.bind(dataServiceName().c_str());
}
/**
 * startReader
 *    Start the reader thread.  The reader is started with an entry point
 *    of readerEntry (static method).  this is passed as the parameter to the
 *    entry point.
 */
void
CRingThreadedBlockReader::startReader()
{
    int status = pthread_create(&m_readerThreadId, nullptr, readerEntry, this);
    if(status != 0) {
        throw std::system_error(
            std::make_error_code(static_cast<std::errc>(errno)),
            "CRingThreadBlockReader::startReader - reader thread start failed"
        );
    }
}
/**
 * shutdown
 *    Called when we've received an end file.
 *    - Send the done message to the reader thread.
 *    - Sockets will close on destruction of this.
 *    - Child sockets will destroy as the threads try to exit.
 *    - join the child thread.
 */
void
CRingThreadedBlockReader::shutdown()
{
    sendDone();
    if (pthread_join(m_readerThreadId, nullptr) != 0) {
        throw std::system_error(
            std::make_error_code(static_cast<std::errc>(errno)),
            "CRingThreadBlockReader::shutdown - joining with reader thread exit"
        );
    }
}
/**
 * processDataMessage
 *    Called when a data message has been received:
 *    - Make a new output buffer that is the size of the input buffer
 *      and any partial item.
 *    - Copy the partial item, its continuation and all the full items
 *      into the new buffer
 *    - Invoke savePartial item to save any partial item in the received buffer.
 *    - Free the received buffer.
 *    - Update the data descriptor as follows:
 *      *  Set the buffer pointer to our new pointer.
 *      *  Set the size to the new size.
 *      *  Set the number of complete items it contains.
 *
 * @param desc - reference to the data descrptor received from the reader.
 * @note - errors in allocation are managed by throwing std::system_error
 * @todo - If the partial item is < 32 bits wide this will fail gloriously.
 */
void
CRingThreadedBlockReader::processDataMessage(DataDescriptor& desc)
{
    // Actually, there are two cases.  If there's no
    // partial item we don't have to do data movement.
    // We'll simplify the data movement case by transferring the
    // entire data buffer if needed.  Then we can just process that:
    
    if (m_partialItemSize) {
        char* pNewBuffer =
            reinterpret_cast<char*>(malloc(m_partialItemSize + desc.s_nBytes));
        
        // Put the partial item into the new buffer:
        
        memmove(pNewBuffer, m_pPartialItem, m_partialItemSize);
        
        // Transfer the read buffer:
        
        memmove(pNewBuffer + m_partialItemSize, desc.s_pData, desc.s_nBytes);
        
        // free the original buffer and fix up the descriptor:
        
        free(desc.s_pData);
        desc.s_pData = pNewBuffer;
        desc.s_nBytes += m_partialItemSize;
        
    }
    // Now we can look at whatever buffer we're using and figure out
    // how many complete items we have, if there's a partial item and,
    // if so save it.
    
    std::uint32_t* pItem;                // To get the ring item size.
    std::uint8_t*  pData =               // To easily step through the buffer
        reinterpret_cast<std::uint8_t*>(desc.s_pData); 
    std::uint32_t  remaining(desc.s_nBytes);            // Bytes processed.
    std::uint32_t  nUsed(0);
    while(1) {
        pItem = reinterpret_cast<std::uint32_t*>(pData);
        std::uint32_t itemSize = *pItem;
        if (itemSize > remaining) {
            // This is a partial item....
            
            savePartialItem(pItem, remaining);
            break;
        } else {
            // This is a full item:
            
            desc.s_nItems++;
            remaining -= itemSize;
            pData +=  itemSize;
            nUsed += itemSize;
        }
    }
    desc.s_nBytes = nUsed;
}
/**
 * serviceName
 *    Returns the name of a service.
 *    Service names consist of a functional prefix (e.g. "credit") with
 *    the file descriptor appropriately appended.  Given one of those functional
 *    prefixes, this method produces the service URL for that service.
 *
 *  @param prefix = service prefix.
 *  @return service URI
 *  @note we assume this is an inproc URI.
*/
std::string
CRingThreadedBlockReader::serviceName(const char* prefix) const
{
    std::stringstream service;
    service << "inproc://" << prefix << ":" << m_nFd;
    return service.str();
}
/**
 * Send a 'done' message to the credit socket.
 *  This tells the reader to exit.
 *
 */
void
CRingThreadedBlockReader::sendDone()
{
    s_send(m_credit, "done");
}
/*--------------------------------------------------------------------------
 * Implementation of Reader (inner class)
 * Note that the implementation of readerEntry is here as that's the
 * logical place for it.
 */


/**
 * CRingThreadedBlockReader::readerEntry
 *    Entry point for the reader thread.
 *    - Establish the object context.
 *    - Construct the reader object.
 *    - Pass control to it.
 *    - return 0 on exit of the Reader.
 *
 * @param args  - thread paramters.  In this case it's actually the
 *                pointer to the CRingFileBlockReader object that wants a
 *                reader thread started.
 */
void*
CRingThreadedBlockReader::readerEntry(void* args)
 {
    CRingThreadedBlockReader* parent  =
        reinterpret_cast<CRingThreadedBlockReader*>(args);
    
    Reader reader(
        parent->m_nFd, parent->m_zmqContext, parent->m_nBlocksize, *parent
    );
    
    reader();
    
    
    pthread_exit(nullptr);
    return nullptr;                // exit the thread.
 }
 
 /**
  * CRingThreadedBlockReader::Reader - constructor.
  *    Save the data and setup the sockets.
  *
  *    @param nFd    - The file descriptor to read.
  *    @param ctx    - The ZMQ socket to useto make the sockets.
  *    @param blockSize - Size of the reads to perform.
  *    @param parent  - Refers to the parent process class.  This allows us
  *                     to make use of some of the services of that class.
  */
 CRingThreadedBlockReader::Reader::Reader(
    int nFd, zmq::context_t& ctx, size_t blockSize,
    CRingThreadedBlockReader& parent
 )  :
    m_nFd(nFd), m_zmqContext(ctx),
    m_credit(ctx, ZMQ_PULL), m_data(ctx, ZMQ_PUSH),
    m_nBlockSize(blockSize)
{
    setupSockets(parent);               // Connect to parent.       
}
 
/**
 * destructor - I think this is empty for now.
 */
CRingThreadedBlockReader::Reader::~Reader(){}

/**
 * operator()
 *     This is the main loop of the thread.
 *     It runs something like this:
 *     getCredits - if credits indicates a done, then return.
 *     Push the number of reads requested by the credit level.
 */
void
CRingThreadedBlockReader::Reader::operator()()
{
    while (1) {
        int nReads = getCredits();
        if (nReads < 0) break;            // Handle unanticipated "done" correctly.
        
        for (int i =0; i < nReads; i++) {
            void* pData = malloc(m_nBlockSize);    // Data blocks are dynamic.
            CRingThreadedBlockReader::DataDescriptor d;
            
            int n = ::read(m_nFd, pData, m_nBlockSize);
            
            assert (n >=0);                      // for now errors blow us up.
            
            if (n) {
                d.s_pData = pData;
                d.s_nBytes = n;
                d.s_nItems = 0;
                sendData(&d);
            } else {
                sendEof();
                return;                        // Eof flushes.
            }
        }
    }
}
/**
 * sendData
 *    Send a data block to the parent.  This is not done zero copy because
 *    the bulk data is not transmitted thanks to address space sharing.
 *
 * @param pData - pointer to the data descriptor.
 */
void
CRingThreadedBlockReader::Reader::sendData(
    CRingThreadedBlockReader::DataDescriptor* pData)
{
    s_sendmore(m_data, "data");                 // Message type.
    zmq::message_t dataMessage(sizeof(CRingThreadedBlockReader::DataDescriptor));
    memcpy(
        dataMessage.data(), pData,
        sizeof(CRingThreadedBlockReader::DataDescriptor)
    );
    m_data.send(dataMessage, 0);             // Message payload.
}

/**
 * sendEof
 *    Send an eof message to the data socket.
 *    We also absorb credit messages until we have a done:
 */
void
CRingThreadedBlockReader::Reader::sendEof()
{
    s_send(m_data, "eof");
    while (getCredits() > 0)
        ;
    // When we land here we got the "done"
}

/**
 * getCredits
 *    Get a credit message from the m_credit socket.
 *
 * @return int - number of credits received.
 * @retval -1  - Got a "done" message.
 */
int
CRingThreadedBlockReader::Reader::getCredits()
{
    std::string type = s_recv(m_credit);
    if (type == "done") return -1;
    
    zmq::message_t creditMsg;
    m_credit.recv(&creditMsg);
    int result;
    memcpy(&result, creditMsg.data(), sizeof(int));
    return result;
}
/**
 * setupSockets
 *    Connect to the server (parent) sockets.
 *
 * @param parent - parent object. This has nice service name computing
 *                 methods that we'll use so we know what to connect with.
 */
void
CRingThreadedBlockReader::Reader::setupSockets(CRingThreadedBlockReader& parent)
{
    std::string service = parent.creditServiceName();
    m_credit.connect(service.c_str());
    
    service = parent.dataServiceName();
    m_data.connect(service.c_str());
}