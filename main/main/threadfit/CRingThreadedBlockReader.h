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

/** @file:  CRingThreadedBlockReader.h
 *  @brief: Send ring items using threading to overlap I/O and
 *          ring boundary computation.l
 *          
 */

#ifndef   CRINGTHREADEDBLOCKREADER_H
#define   CRINGTHREADEDBLOCKREADER_H
#include <pthread.h>
#include <zmq.hpp>

/**
 * @class CRingThreadedBlockReader
 *
 *    This class overlaps synchronous I/O for fixed sized block reads with the
 *    ring item logic in CRingFileBlockReader.h.   It does so by spawning off
 *    a thread that does credit based I/O.   We use a pair of Push/Pull sockets
 *    to manage this.  The 'credit' socket is used to PUSH read credits from
 *    the parent thread to the child thread.  The child thread reads the requested
 *    number of blocks from file and PUSHes them through the second socket to
 *    the parent which then does the needed work to determine the number of
 *    ring items in each block and passes them to the caller.  To the caller, this
 *    class provides the same blocking interface as CRingFileBlockReader.
 *
 *    As usual, a bit of care is needed to handle end conditions.  In order for
 *    the threads to properly exit there can be no inflight messages.  To see how
 *    this will be managed, let's first look at the structure of the messages
 *    exchanged.  The Credit push/pull pair has messages pushed by the parent
 *    process to the child.  Two types of messages can be pushed:
 *
 *    Credit messages Boxes below are message parts:
 *    \verbatim
 *                 +------------------------------+
 *                 |   "credit"                   |
 *                 +------------------------------+
 *                 |  Integer number of credits   |
 *                 +------------------------------+
 *    \endverbatim
 *
 *    Done messages:
 *    \verbatimn
 *                +------------------------------+
 *                |   "done"                     |
 *                +------------------------------+
 *     \endverbatim
 *    The data push/pull messagess, similarly can transport two types of messages:
 *
 *    Data messages:
 * \verbatim
 *               +------------------------------+
 *               |    "data"                    |
 *               +------------------------------+
 *               | DataDescriptor               | 
 *               +------------------------------+
 *  \endverbatim
 *
 *  Note the nbytes of the data descriptor is filled in with the actual number
 *  of bytes read. That value will always be > 0. see below.
 *
 *    End file messages:
 *
 *    \verbatim
 *              +------------------------------+
 *              |  "eof"                       |
 *              +------------------------------+
 *    \endverbatim
 *
 *
 *  Here's how communication goes.
 *  The parent process  sends off a set of credits.  This prompts the reader(child)
 *  to read a bunch of blocks and push them as data to the parent.  As those blocks
 *  are processed the parent pushes more credits.
 *
 *  At some point, the reader hits an end file.  When that happens, it pushes an 'eof'
 *  message.  Meanwhile there may be many data blocks and credit messages in flight.
 *  The parent continues to pull data blocks until it encounters the 'eof' at which
 *  time it pushes a 'done' and then closes both sockets. and joins the reader thread.
 *   Once the reader sees the
 *   end file, it continues to read from the credit pipe, ignoring messages until
 *   it sees the 'done' at which time it closes its sides of both sockets and exits.
 *
 *    I believe this application level protocol should properly terminate both
 *    sockets.
 */
class CRingThreadedBlockReader {
    
    // Note that all of this data is for the parent process.
    
private:
    pthread_t       m_readerThreadId;
    zmq::context_t& m_zmqContext;
    zmq::socket_t    m_credit;
    zmq::socket_t    m_data;
    size_t          m_nBlocksize;
    
    size_t          m_nCredits;
    size_t          m_nCreditsRemaining;
    size_t          m_nCreditWaterMark;
    
    int m_nFd;
    std::uint32_t  m_partialItemSize;       // How much data is in the partial item.
    std::uint32_t  m_partialItemBlockSize;	// How big is the buffer pointed to by m_pPartialItem.
    char* m_pPartialItem;

    bool          m_fEOF;
public:
    typedef struct _DataDescriptor {
      std::uint32_t s_nBytes;
      std::uint32_t s_nItems;
      void*         s_pData;
    } DataDescriptor, *pDataDescriptor;
  
    // Public interface:
    
public:
    CRingThreadedBlockReader(
        const char* file, zmq::context_t& ctx, size_t bytes,
        size_t credits, size_t creditWaterMark
    );
    
    virtual ~CRingThreadedBlockReader();
    
    DataDescriptor read();                  // get next block of complete ring items.
    std::string creditServiceName() const;
    std::string dataServiceName()   const;
    
    
    // This nested class is the reader thread:
    
    
    class Reader {
    private:
        int             m_nFd;            // File descriptor
        zmq::context_t& m_zmqContext;
        zmq::socket_t   m_credit;
        zmq::socket_t   m_data;
        
        size_t          m_nBlockSize;
    public:
        Reader(
            int nFd, zmq::context_t& ctx, size_t blockSize,
            CRingThreadedBlockReader& parent
        );
        ~Reader();
        
        void operator()();               
    private:
        void sendData(DataDescriptor* pData);
        void sendEof();
        int  getCredits();
        void setupSockets(CRingThreadedBlockReader& parent);
    };
    
    
    // Private utilities:
private:
    void savePartialItem(void* pItem, size_t nBytes);
    void sendCredits(int num);
    DataDescriptor readData();
    void setupSockets();
    void startReader();
    void shutdown();
    void processDataMessage(DataDescriptor& desc);
    std::string serviceName(const char* baseName) const;
    void sendDone();
    // Thread starter:

private:
    static void* readerEntry(void* args);
};


#endif


