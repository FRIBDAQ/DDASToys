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

/**
 * @file  RootExtensions.h
 * @brief Provide streamable versions of the event extensions defined in
 * fit_extensions.h
 *
 * @details
 * These structs mirror those having to do with the fit extensions to DDAS 
 * data, but they support ROOT serialization and, therefore, can be put into
 * ROOT files/tree leaves.
 */

#ifndef ROOTEXTENSIONS_H
#define ROOTEXTENSIONS_H

#include <fit_extensions.h>
#include <TObject.h>

/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootPulseDescription
 * @brief Describes a single pulse without an offset.
 */
struct RootPulseDescription : public DDAS::PulseDescription, public TObject
{  
    /** @brief Required for inheritence from TObject. */  
    ClassDef(RootPulseDescription, 1);
};

/** @} */


/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootFit1Info
 * @brief Full fitting information for the single pulse.
 */
struct RootFit1Info : public DDAS::fit1Info, public TObject
{
    /** @brief Required for inheritence from TObject. */
    ClassDef(RootFit1Info, 1);
};

/** @} */

/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootFit2Info
 * @brief Full fitting information for the double pulse.
 */
struct RootFit2Info : public DDAS::fit2Info, public TObject
{
    /** @brief Required for inheritence from TObject. */
    ClassDef(RootFit2Info, 1);
};

/** @} */

/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootHitExtension
 * @brief The data structure containing the full information about the fits.
 */
struct RootHitExtension : public DDAS::HitExtension, public TObject
{
    /** @brief Required for inheritence from TObject. */
    ClassDef(RootHitExtension, 1);
};

/** @} */

#endif
