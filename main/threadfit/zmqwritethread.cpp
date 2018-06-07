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
#include <CFileDataSink.h>
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
 *      |  thread-id (uint32_t)        |
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
 *      | thread-id (uint32_t)          |
 *      +-------------------------------+
 *
 *    Indicates we should not expect any more data from  node-id.

 */

/**
 * Constants:
 */

static const int PORT(5672);        // Port part of URIs.

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
    std::stringstream uri;
    uri << "tcp://*:" << PORT;
    
    return uri.str();
}
/**
 * ClientUri
 *    @return std::string - returns the URI the client should connect to.
 */
static std::string
ClientUri()
{
    std::stringstream uri;
    uri << "tcp://localhost:" << PORT;
    return uri.str();
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
    pSock->connect(ClientUri().c_str());
    
    return *pSock;
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
 * sendEnd
 *    Send end of data moarker.
 *
 * @param sock - the socket on which to send the message.
 * @param thread - Thread id of  the sender.
 */
void
sendEnd(zmq::socket_t& sock, int thread)
{
    sendHeader(sock, "end", thread, false);
}

/*------------------------------------------------------------------------------
 *  The actual writer thread
 */

/**
 * zmqwriter_thread
 *    Thread entry point
 *
 * @param args - actually a pointer to an array of character pointers to the
 *                filename and file format.
 * @return void* - nothing actually.
 */
void*
zmqwriter_thread(void* args)
{
    // Listen on our socket.
    
    zmq::context_t context;
    zmq::socket_t  sock(context, ZMQ_PULL);    // As a fan in.
    int linger(0);
    sock.setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    sock.bind(ServerUri().c_str());
    
    //  Set up the File name data sink with appropriate format:
    
    char** pArgs = reinterpret_cast<char**>(args);
    std::string filename   = pArgs[0];
    std::string fileFormat = pArgs[1];

    /** @todo - create the data sinktype based on the fileFormat param */
    

    CFileDataSink outFile(filename);
    
    
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
    
    std::set<int>   activeThreads;   
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
            CRingItem** pItem = static_cast<CRingItem**>(ring.data());
            outFile.putItem(**pItem);
            delete *pItem;              // We required this be dynamic.
            
        } else if (type == "end") {
            // Take the node out of the set.  If the resulting set is empty,
            // we're done.
            
            activeThreads.erase(id);
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
