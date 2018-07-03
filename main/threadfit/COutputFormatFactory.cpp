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

/** @file:  COutputFormatFactory.cpp
 *  @brief: Given and output format and a connection string (e.g. filename)
 *          Returns a COutputSink of the appropriate type.
 */
#include "COutputFormatFactory.h"
#include "CRootSelectableDataSink.h"
#include "CRootFileDataSink.h"
#include "CFileDataSink.h"

#include <stdexcept>

/**
 * createSink
 *    Static method to create a new output sink.
 *
 *  @param format - type of sink to make, e.g. 'ring' or 'root'.
 *  @param connection - What the ring should connect to e.g. a filename.
 *  @return COutputSink* - Pointer to a new'd output sink object
 *                         of the type and connection requested.
 *  @throws std::invalid_argument - if format is invalid or anything one of the
 *                        output sink class constructors might throw.
 */
CDataSink*
COutputFormatFactory::createSink(const char* format, const char* connection)
{
    std::string fmt(format);
    
    if (fmt == "ring") {
        return new CFileDataSink(connection);
    } else if (fmt == "root") {
        return new CRootFileDataSink(connection);
    } else if (fmt == "rootselectable") {
        return new CRootSelectableDataSink(connection);
    } else {
        throw std::invalid_argument("Invalid output format type.");
    }
}