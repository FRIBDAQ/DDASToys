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

/** @file:   zmqwritethread.cpp
 *  @brief:  Implement thread to write data from workers to disk.
 */


#include "zhelpers.hpp"
#include "CSortingOutputter.h"

#include <CDataSink.h>
#include "COutputFormatFactory.h"


#include <CRingItem.h>

#include <string>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdint.h>
#include <set>
#include <stdlib.h>

/**
 * In this code we implement both the API for workers to queue data
 * for output to the outpu thread and the output thread itself.
 *
 * The output thread is a server the workers are clients.  The output thread
 * PULL's data from the clients which PUSH it.
 *
 *  Messages have two or three parts.  There are two types of messages:
 *
 *  Data message:
 *      +------------------------------+
 *      |   "data"                     |
 *      +------------------------------+
 *      |  node-id (uint32_t)          |
 *      +------------------------------+
 *      | CRingItem*                   |
 *      +------------------------------+
 *
 *      In data messages CRingItem* points to a dynamically allocated ring
 *      item that must be 'delete'd by the writethread once it's been written
 *      to disk.
 *  End message:
 *      +-------------------------------+
 *      |  "end"                        |
 *      +-------------------------------+
 *      | node-id (uint32_t)            |
 *      +-------------------------------+
 *
 *    Indicates we should not expect any more data from  node-id.
 *
*
 *    Each thread must register its existence before any thread sends data.
 *    For now the assumption is that so much time is required to analyze a block
 *    of data that no synchronization is required for this.  If that turns out
 *    not to work, the master thread could do that registration on the worker's
 *    behalf _before_ starting the worker threads.
 *    The registration message looks like this:
 *
 *    +---------------------------------+
 *    |   "reg"                         |
 *    +---------------------------------+
 *    |   thread-id                     |
 *    +---------------------------------+
 *
 *    This registration is used to build a queue for data received from the
 *    thread so that all data can be time ordered.
 */

/**
 * Constants:
 */


static  int MAX_QUEUED_MESSAGES(128);

/*-----------------------------------------------------------------------------
 *   Internal utility methods.
 */

/**
 *   ServerUri:
 *   
 *   @return std::string - returns the URI the server should bind to:
 */
static std::string
ServerUri()
{
    return std::string("inproc://writer");
    
}
/**
 * ClientUri
 *    @return std::string - returns the URI the client should connect to.
 */
static std::string
ClientUri()
{
    return std::string("inproc://writer");
}

/**
 * RegServerUri
 *    Returns the registration URI for the server.
 */
static std::string
RegServerUri()
{
    return std::string("inproc://registrar");
 
}
/**
 * RegClientUri
 *   Return the registration URI for the client.
 */
static std::string
RegClientUri()
{
    return std::string("inproc://registrar");
    
}

/**
 * sendHeader
 *    Send the message header (type and sourceid) to the writer:
 *
 * @param sock   - Reference to the socket used to transport the data.
 * @param type   - type string.
 * @param thread - Source thread
 * @param more   - True if there will be at least one more message part.
 */
static void
sendHeader(zmq::socket_t& sock, const char* type, uint32_t thread, bool more)
{
    s_sendmore(sock, std::string(type));
    
    zmq::message_t id(sizeof(uint32_t));        // thread id.
    memcpy(id.data(), &thread, sizeof(uint32_t));
    sock.send(id, more ? ZMQ_SNDMORE : 0);
    
}
/*-----------------------------------------------------------------------------
 *   Client functions.
 */

/**
 * makeClientSocket
 *     Returns a reference to a dynamically allocated zmq::socket_t that's
 *     connected to the zmqwriter_thread's PULL server.
 *     The socket will be a ZMQ_PUSH socket.
 *
 * @param ctx - reference to the context on which the socket should be created.
 * @return zmq::socket_t& - caller must delete the socket at some point.
*/

zmq::socket_t&
makeClientSocket(zmq::context_t& ctx)
{
    zmq::socket_t* pSock = new zmq::socket_t(ctx, ZMQ_PUSH);
    int linger(0);
    pSock->setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    pSock->setsockopt(ZMQ_SNDHWM, &MAX_QUEUED_MESSAGES, sizeof(int));
    pSock->setsockopt(ZMQ_RCVHWM, &MAX_QUEUED_MESSAGES, sizeof(int));
    pSock->connect(ClientUri().c_str());
    
    return *pSock;
}
/**
 * makeRegistrationSocket
 *    Create the REQ side of the registration REQ/REP pair of sockets
 *    used to register threads with the output thread.
 *
 * @param ctx - references the ZMQ context in which the socket is created.
 * @return zmq::socket_t&  - pointer to a dynamically created socket  that
 *              must at some point be deleted by the caller.
 */
zmq::socket_t&
makeRegistrationSocket(zmq::context_t& ctx)
{
    zmq::socket_t* pSock = new zmq::socket_t(ctx, ZMQ_REQ);
    zmq::socket_t& result(*pSock);
    
    int linger(0);
    result.setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    
    result.connect(RegClientUri().c_str());
    return result;
}


/**
 * sendRingItem
 *    Send a processed ring item to the writer.
 *
 *  @param sock    - reference to a socket returned from makeClientSocket.
 *  @param thread  - Id of the thread that's sending this item.
 *  @param item    - Pointer to the ring item to write
 *  @note The writer will delete 'item' so that must be a pointer to a
 *        dynamically allocated ring item.
 *        
 */
void
sendRingItem(zmq::socket_t& sock, int thread, CRingItem* pItem)
{
    sendHeader(sock, "data", thread, true);
    
    zmq::message_t item(sizeof(CRingItem*));
    memcpy(item.data(), &pItem, sizeof(CRingItem*));
    sock.send(item, 0);
    

    
}
/**
 * endRegistrations
 *    Used to let the server know that we're done registering threads and that
 *    the server can get to the business of receiving data and outputting it:
 *
 *  @param sock - references the socket used to talk to the server.
 */
void
endRegistrations(zmq::socket_t& sock)
{
    sendHeader(sock, "end", 0, false);
    
    zmq::message_t reply;
    sock.recv(&reply, 0);    // Indicates message received.
}
/**
 * sendEnd
 *    Send end of data marker.
 *
 * @param sock - the socket on which to send the message.
 * @param thread - Thread id of  the sender.
 */
void
sendEnd(zmq::socket_t& sock, int thread)
{
    sendHeader(sock, "end", thread, false);
}
/**
 * registerThread
 *    Register a worker thread with the sort/output thread.  See the
 *    comments describing the message structur for more on what and why this
 *    is.
 *
 * @param sock  - reference to the socket used to communicate with the outputter.
 *                Get this via a call to makeClientSocket
 * @param thread - Id of the thread to register.f
 */
void
registerThread(zmq::socket_t& sock, int thread)
{
    sendHeader(sock, "reg", thread, false);
    
    // Accept the reply:
    
    zmq::message_t reply;
    sock.recv(&reply, 0);
}
/*------------------------------------------------------------------------------
 *  The actual writer thread
 */

/**
 * zmqwriter_thread
 *    Thread entry point
 *
 * @param args - actually a pointer to the const char* name of the
 *               file to write.
 * @return void* - nothing actually.
 */
void*
zmqwriter_thread(void* args)
{
    // Listen on our data socket.
    
    zmq::context_t& context(getContext());
    zmq::socket_t  sock(context, ZMQ_PULL);    // As a fan in.
    sock.setsockopt(ZMQ_SNDHWM, &MAX_QUEUED_MESSAGES, sizeof(int));
    sock.setsockopt(ZMQ_RCVHWM, &MAX_QUEUED_MESSAGES, sizeof(int));
    int linger(0);
    sock.setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    sock.bind(ServerUri().c_str());
    
    //  Set up the File data sink.

    char** pArgs = reinterpret_cast<char**>(args);
    std::string filename(pArgs[0]);
    std::string fileFormat(pArgs[1]);


    CDataSink* pSink =
        COutputFormatFactory::createSink(fileFormat.c_str(), filename.c_str());
    
    CDataSink& outFile(*pSink);
    CSortingOutputter outputter(outFile);
    
    // Set up the REQ/REP socket and accept the thread registrations:
    //
    std::set<int>   activeThreads;                 // Set of active/registered threads.
    zmq::socket_t regsock(context, ZMQ_REP);       // We  reply to requests.
    regsock.setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    regsock.bind(RegServerUri().c_str());
    
    // Process registrations until an "end" message
    while(1) {
        // All messages here are two part, type, and socket....event
        // if socket is dummy for the ends:
        
        std::string type = s_recv(regsock);
        zmq::message_t threadMsg;
        regsock.recv(&threadMsg, 0);
        uint32_t thread;
        memcpy(&thread, threadMsg.data(), sizeof(uint32_t));
        
        if (type == "reg") {
            activeThreads.insert(thread);
            outputter.addSource(thread);
            std::string reply("ok");
            s_send(regsock, reply);
            
        } else if (type == "end") {
            // Done with registrations, shut down the socket and break
            // from the loop.
            
            std::string reply("ok");
            s_send(regsock, reply);
            
            break;
        }
    }
    regsock.close();
    
    // Process ring items until there are no more souces.
    // This assumes the first message received is not an end -- it could
    // be for a file with fewer items than workers.
    
    // A bit on how this works. As 'data' messages come in, threads are added to
    // the activeThreads set.  As 'end' messages come in, those threads are removed.
    // once the activeThreads set is empty (after a first message), we're done
    
    // Note again, that if the first message we get is an 'end'...well that
    // blows our logic away.  This logic is predicated on the assumption we're
    // processing _big_ files.  If this were not the case, there's no reason for
    // a parallel implementation.
    
    
    while(1) {
        
        // Get a message type and source:
        
        std::string type = s_recv(sock);
        zmq::message_t msg;
        sock.recv(&msg, 0);
        uint32_t id;
        memcpy(&id, msg.data(), sizeof(uint32_t));
        
        // The rest of the processing is message type dependent:
        
        if (type == "data") {
            
            
            // Receive the ring item pointer.
            
            zmq::message_t ring;
            sock.recv(&ring, 0);
            CRingItem* pItem = *static_cast<CRingItem**>(ring.data());
            outputter.queueItem(id, pItem->getEventTimestamp(), pItem);
            
        } else if (type == "end") {
            // Take the node out of the set.  If the resulting set is empty,
            // we're done.
            
            activeThreads.erase(id);
            outputter.shutdownSource(id);
            if (activeThreads.empty()) {
                break;
            }
            
        } else {
            std::cerr << "zmqwriter_thread - got an unrecognized message type: "
                << type << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    sock.close();
    
    //      Exit.
    
     
    return nullptr;
}
