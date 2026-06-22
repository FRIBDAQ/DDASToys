/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2016.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
	     Aaron Chester
	     Facility for Rare Isotope Beams
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

#include <cppunit/extensions/HelperMacros.h>

#include "Asserts.h"
#include "DDASFitHit.h"

using namespace ddastoys;

class FitHitTests : public CppUnit::TestFixture
{
public:
    CPPUNIT_TEST_SUITE(FitHitTests);

    CPPUNIT_TEST(construct1);
    CPPUNIT_TEST(setget1);
    CPPUNIT_TEST(setget2);
    CPPUNIT_TEST(setget3);
    CPPUNIT_TEST(assign1);
    CPPUNIT_TEST(assign2);
    CPPUNIT_TEST(assign3);
    
    CPPUNIT_TEST_SUITE_END();

public:
    void construct1(); // default
    void setget1();    // set extension data
    void setget2();    // set extension data and read it
    void setget3();    // error if getting extension when there is none
    void assign1();    // copy assigment w/o extension
    void assign2();    // copy construct w/ extension
    void assign3();    // copy construct w/ extension and check the data
};

void FitHitTests::construct1()
{
    DDASFitHit hit;
    
    EQMSG("Default construction w/o extension", false, hit.hasExtension());
}

void FitHitTests::setget1()
{
    DDASFitHit hit;
    HitExtension ext;
    hit.setExtension(ext);

    EQMSG("Set the fit extension data", true, hit.hasExtension());
}

void FitHitTests::setget2()
{
    DDASFitHit hit;
    HitExtension set;
    set.onePulseFit.iterations = 1234;
    hit.setExtension(set);
    auto get = hit.getExtension();
    
    EQMSG("Get data from set extension",
	  (unsigned)1234, get.onePulseFit.iterations);
}

void FitHitTests::setget3()
{
    DDASFitHit hit;
    EXCEPTION(hit.getExtension(), std::logic_error);
}

void FitHitTests::assign1()
{
    DDASFitHit hit1;
    DDASFitHit hit2 = hit1;
    
    EQMSG("Copy assignment w/o extension", false, hit2.hasExtension());
}

void FitHitTests::assign2()
{
    DDASFitHit hit1;
    HitExtension ext;
    hit1.setExtension(ext);
    DDASFitHit hit2 = hit1;
    
    EQMSG("Copy assigment w/ extension", true, hit2.hasExtension());
}

void FitHitTests::assign3()
{
    DDASFitHit hit1;
    HitExtension set;
    set.onePulseFit.iterations = 1234;
    hit1.setExtension(set);

    DDASFitHit hit2 = hit1;
    auto get = hit2.getExtension();
    
    EQMSG("Get data from assigned extension",
	  (unsigned)1234, get.onePulseFit.iterations);
}


CPPUNIT_TEST_SUITE_REGISTRATION(FitHitTests);
