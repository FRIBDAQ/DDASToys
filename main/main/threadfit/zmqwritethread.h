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

/** @file  zmqwritethread.h
 *  @brief Receive ring items from worker threads and output to file.
 */

#ifndef ZMQWRITERTHREAD_H
#define ZMQWRITERTHREAD_H
#include <zmq.hpp>

class CRingItem;



zmq::socket_t& makeClientSocket(zmq::context_t& ctx);
void sendRingItem(zmq::socket_t& sock, int thread,  CRingItem* item);
void sendEnd(zmq::socket_t& sock, int thread);

extern void* zmqwriter_thread(void* args);

#endif