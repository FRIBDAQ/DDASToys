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

#include <DataFormat.h>

#include "Asserts.h"
#include "DDASFitHit.h"
#include "DDASFitHitUnpacker.h"

using namespace ddastoys;
using namespace ufmt;

/**
 * @note (ASC 1/5/26): DDASFitHitUnpacker is derived from DDASHit unpacker,
 * which has its own test suite to unpack the actual hit data. We assume that
 * works and only test unpacking hits with various extensions.
 */
class FitUnpackerTests : public CppUnit::TestFixture
{
public:
    DDASFitHit hit;
    DDASFitHitUnpacker unpacker;
    std::vector<std::uint32_t> data; // Pixie payload data
    RingItem* pItem; // hit as an FRIBDAQ ring item
    size_t dataSize;
    
public:
    CPPUNIT_TEST_SUITE(FitUnpackerTests);
    
    CPPUNIT_TEST(unpack1);
    CPPUNIT_TEST(unpack2);
    CPPUNIT_TEST(decode1);
    CPPUNIT_TEST(decode2);
    
    CPPUNIT_TEST_SUITE_END();
    
public:
    void setUp()
	{
	    data = {
		0x00000030, 0x00fa0f10, 0x002d2321, 0x0000f687,
		0x547f000a, 0x000808be,	0x00000001, 0x00000002,
		0x00000003, 0x00000004, 0x00000005, 0x00000006,
		0x00000007, 0x00000008, 0x00000009, 0x0000000a,
		0x0000000b, 0x0000000c, 0x0a0a0b0b, 0x0c0c0d0d,
		0x00020001, 0x00040003, 0x00060005, 0x00080007
	    };
	    unpacker.unpack(data.data(), data.data() + data.size(), hit);
	    dataSize = data.size()*sizeof(uint32_t);
	    //pItem = (RingItem*)malloc(sizeof(RingItem));
	};

    void tearDown()
	{
	    free(pItem);
	}
    
    void unpack1(); // unpack raw DDASHit
    void unpack2(); // unpacked raw data doesn't have extension
    void decode1(); // decode ring item with no extension
    void decode2(); // decode ring item with extension
    void decode3(); // decode ring item with legacy extension
    void decode4(); // decode ring item with null extension
};

void FitUnpackerTests::unpack1()
{
    EQMSG("Unpack correct crate ID", uint32_t(3), hit.getCrateID());
}

void FitUnpackerTests::unpack2()
{
    EQMSG("Unpacked event has no extension", false, hit.hasExtension());
}

void FitUnpackerTests::decode1()
{
    uint32_t bodySize = data.size() * sizeof(uint32_t);
    uint32_t totalSize = sizeof(RingItemHeader) + sizeof(BodyHeader)
	+ bodySize;
        
    pItem = (RingItem*)realloc(pItem, totalSize);    
    pItem->s_header = {totalSize, PHYSICS_EVENT};
    pItem->s_body.u_hasBodyHeader.s_bodyHeader = {
	sizeof(BodyHeader), 1234, 0, 0
    };
    memcpy(pItem->s_body.u_hasBodyHeader.s_body, data.data(), bodySize);

    DDASFitHit myhit;
    unpacker.decode(pItem, myhit);

    EQMSG("Decode correct crate ID from raw hit",
	  hit.getCrateID(), myhit.getCrateID());
}

void FitUnpackerTests::decode2()
{
    uint32_t bodySize = dataSize + sizeof(FitInfo);
    uint32_t totalSize = sizeof(RingItemHeader) + sizeof(BodyHeader)
	+ bodySize;
    FitInfo fit;
    
    pItem = (RingItem*)realloc(pItem, totalSize);    
    pItem->s_header = {totalSize, PHYSICS_EVENT};
    pItem->s_body.u_hasBodyHeader.s_bodyHeader = {
	sizeof(BodyHeader), 1234, 0, 0
    };
    memcpy(pItem->s_body.u_hasBodyHeader.s_body, data.data(), dataSize);
    memcpy(pItem->s_body.u_hasBodyHeader.s_body + dataSize,
	   &fit, sizeof(FitInfo));

    DDASFitHit myhit;
    unpacker.decode(pItem, myhit);
    
    EQMSG("Decoded fit hit has extension",
	  true, myhit.hasExtension());
    EQMSG("Decode correct crate ID from fit hit",
	  hit.getCrateID(), myhit.getCrateID());    
}

void FitUnpackerTests::decode3()
{
    uint32_t bodySize = dataSize + sizeof(FitInfoLegacy);
    uint32_t totalSize = sizeof(RingItemHeader) + sizeof(BodyHeader)
	+ bodySize;
    FitInfoLegacy fit;
    
    pItem = (RingItem*)realloc(pItem, totalSize);    
    pItem->s_header = {totalSize, PHYSICS_EVENT};
    pItem->s_body.u_hasBodyHeader.s_bodyHeader = {
	sizeof(BodyHeader), 1234, 0, 0
    };
    memcpy(pItem->s_body.u_hasBodyHeader.s_body, data.data(), dataSize);
    memcpy(pItem->s_body.u_hasBodyHeader.s_body + dataSize,
	   &fit, sizeof(FitInfoLegacy));

    DDASFitHit myhit;
    unpacker.decode(pItem, myhit);
    
    EQMSG("Decoded legacy fit hit has extension",
	  true, myhit.hasExtension());
    EQMSG("Decode correct crate ID from legacy hit",
	  hit.getCrateID(), myhit.getCrateID());    
}

void FitUnpackerTests::decode4()
{
    uint32_t bodySize = dataSize + sizeof(nullExtension);
    uint32_t totalSize = sizeof(RingItemHeader) + sizeof(BodyHeader)
	+ bodySize;
    nullExtension nullext;
    
    pItem = (RingItem*)realloc(pItem, totalSize);    
    pItem->s_header = {totalSize, PHYSICS_EVENT};
    pItem->s_body.u_hasBodyHeader.s_bodyHeader = {
	sizeof(BodyHeader), 1234, 0, 0
    };
    memcpy(pItem->s_body.u_hasBodyHeader.s_body, data.data(), dataSize);
    memcpy(pItem->s_body.u_hasBodyHeader.s_body + dataSize,
	   &nullext, sizeof(nullExtension));

    DDASFitHit myhit;
    unpacker.decode(pItem, myhit);
    
    EQMSG("Decoded null fit hit has extension",
	  true, myhit.hasExtension());
}

CPPUNIT_TEST_SUITE_REGISTRATION(FitUnpackerTests);
