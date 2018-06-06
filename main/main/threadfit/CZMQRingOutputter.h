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

/** @file:  CZMQRingOutputter.h
 *  @brief: Outputter for CDDASAnalyzer that sends ring item pointers to a zmq socket.
 */
#ifndef CZMQRINGOUTPUTTER_H
#define CZMQRINGOUTPUTTER_H
#include "Outputter.h"
#include <zmq.hpp>
#include <FragmentIndex.h>

class CRingItem;

/**
 * @class CZMQRingOutputter
 *     Derived from Outputter.  Uses the API in zmqwritethread to send
 *     data to a writer thread.  An instance of this should be in each
 *     worker thread to send data to be serially/synchronously written to the
 *     output file.
 */
class CZMQRingOutputter : public Outputter
{
private:
    zmq::socket_t& m_socket;
    
public:
    CZMQRingOutputter(zmq::socket_t& sock);
    virtual ~CZMQRingOutputter() {}
    
    virtual void outputItem(int id, void* pItem);
    virtual void end(int id);
private:
    void deleteIfNeeded(FragmentInfo& frag);

};

#endif