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

/** @file:  Outputter.h
 *  @brief: Abstract base class for outputters that are attached to analyzers..
 */
#ifndef OUTPUTTER_H
#define OUTPUTTER_H

/**
 * @class Outputter
 *    Analysis classes need some way to pass their data on to whatever is next.
 *    This is done by attaching an Outputter to them.
 *    The Outputter is given the results of that analyzer to pass on to
 *    whatever comes next.  It's anticpated that concrete outputters may
 *    well be quite special purpose entities as the analyis output format may
 *    vary considerably from analyzer to analyzer as what to do with it as well.
 */
class Outputter
{
public:
    virtual ~Outputter() {}                 // Establish virtual destruction.
    
    virtual void outputItem(int id, void* pItem) = 0;  // Output a unit of data.
    virtual void end(int id) {}             // Perform any end of data notification
};

#endif