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

/** @file:  COutputFormatFactory.h
 *  @brief: Create the proper output sink given a format and sinkname.
 */

#ifndef COUTPUTFORMATFACTORY_H
#define COUTPUTFORMATFACTORY_H

class COutputSink;

/**
 * @class COutputFormatFactory
 *    This factory produces the appropriate CDataSink object given
 *    - format type.
 *    - Connection name (filename e.g.).
 *
 *    It's being supplied because of the proliferation of output formats
 *    for the ringblockdealer fitter.
 */
class COutputFormatFactory {
    static COutputSink* createSink(const char* format, const char* connection);
}

#endif