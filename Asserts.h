/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2016.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Ron Fox
	     Jeromy Tompkins
	     Facility for Rare Isotope Beams
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

#ifndef __ASSERTS_H
#define __ASSERTS_H

#include <iostream>
#include <string>

// Abbreviations for assertions in cppunit.

#define EQMSG(msg, a, b)     CPPUNIT_ASSERT_EQUAL_MESSAGE(msg, a, b)
#define EQ(a, b)             CPPUNIT_ASSERT_EQUAL(a, b)
#define ASSERT(expr)         CPPUNIT_ASSERT(expr)
#define ASSERTMSG(msg, expr) CPPUNIT_ASSERT_MESSAGE(msg, expr)
#define FAIL(msg)            CPPUNIT_FAIL(msg)

// Macro to test for exceptions:

#define EXCEPTION(operation, type)		\
    {						\
	bool ok = false;			\
	try {					\
	    operation;				\
	}					\
	catch (const type& e) {			\
	    ok = true;				\
	}					\
	ASSERT(ok);				\
    }

class Warning {

public:
    Warning(std::string message) {
	std::cerr << message << std::endl;
    }
};


#endif
