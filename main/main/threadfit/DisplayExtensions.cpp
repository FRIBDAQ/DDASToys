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

/** @file:  DisplayExtensions.cpp
 *  @brief: Show info in extensions for DDAS Ring item analyzed file (output from TestAnalyzer.cpp).
 */

#include <CFileDataSource.h>
#include <FragmentIndex.h>
#include <CRingItem.h>
#include "FitHitUnpacker.h"
#include "DDASFitHit.h"
#include "functions.h"
#include <URL.h>
#include <Exception.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <stdlib.h>



/**
 *  Usage:
 *    DispalyExtensions infile
 * Where:
 *    infile - the name of a ring item file that contains DDAS Data.
 *
 * For each item that has a waveform with a Hit extension:
 *  - Outputs the fit parameters of both fits.
 *  - Creates/displays a plot containing the raw data and the
 *    two fit lines.  This is done with diffplot - not the right
 *    thing but it's handy and displays three trace-lines.
 */


/**
 * usage
 *    Print error message and usage info to a stream.
 *  @param o - stream the data are output to.
 *  @param msg - the error message.
 */
static void
usage(std::ostream& o, const char* msg)
{
    o << msg << std::endl;
    o << "Usage:  \n";
    o << "    DisplayExtensions infile\n";
    o << "Where:\n";
    o << "   infile is the name of a file with ring items\n";
    exit(EXIT_FAILURE);
}

/**
 * filenameToUri
 *    Constructs the file URI that's assocaiated with the filename.
 *
 * @param name - name of the file to URI-ize.
 * @return URL - object wrapping the URI.
 */
static URL
filenameToUri(const char* name)
{
    char* absoluteName = canonicalize_file_name(name);
    std::string uriString = "file://";
    uriString += absoluteName;
    URL result(uriString);
    free(absoluteName);
    
    return result;
}
/**
 * plotFit
 *    Write the trace, and fits for both single an double pulse fits
 *    to file and spin off diffplot to display them.
 *
 *  @param trace - reference to the trace.
 *  @param ext   - The fit extension with fit information.
 */
 static void
 plotFit(const std::vector<uint16_t>& trace, const ::DDAS::HitExtension& ext)
 {
    std::ofstream o("realdata.dat");      // Couldn't think of a good name.
    
    // For convenience extract the parameters:
    
    // Fit 1:
    
    double A0 = ext.onePulseFit.pulse.amplitude;
    double X0 = ext.onePulseFit.pulse.position;
    double K1 = ext.onePulseFit.pulse.steepness;
    double K2 = ext.onePulseFit.pulse.decayTime;
    double C0 = ext.onePulseFit.offset;
    
    // Fit 2:
    
    double A20 =  ext.twoPulseFit.pulses[0].amplitude;
    double A21 =  ext.twoPulseFit.pulses[1].amplitude;
    double X20 =  ext.twoPulseFit.pulses[0].position;
    double X21 =  ext.twoPulseFit.pulses[1].position;
    double K21 =  ext.twoPulseFit.pulses[0].steepness;
    double K23 =  ext.twoPulseFit.pulses[1].steepness;
    double K22 =  ext.twoPulseFit.pulses[0].decayTime;
    double K24 =  ext.twoPulseFit.pulses[1].decayTime;
    double C2  =  ext.twoPulseFit.offset;
    
    for (int i = 0; i < trace.size(); i++) {
        o << i <<  " " << trace[i] << " " 
            << DDAS::singlePulse(A0, K1, K2, X0, C0, i) << " "
            << DDAS::doublePulse(
                A20, K21, K22, X20,
                A21, K23, K24, X21, C2, i
            ) << std::endl;
    }
    
    o.close();
    system("diffplot realdata.dat");
 }
/**
 * reportHitFits
 *    Report fit information from a hit. Note that the hit is assumed
 *    to have an extension already.
 *    - Report the fit result to cout.
 *    - Create a plot file for the trace and two fits, use diffplot to plot them.
 *    
 *  @param hit - reference to the hit to report.
 *  @note there's no fit for hits that don't have a trace so we can assuume
 *        there's a trace.
 */
static void
reportHitFits(const DAQ::DDAS::DDASFitHit& hit)
{
    // Get the stuff we need:
    
    const std::vector<uint16_t>& trace(hit.GetTrace());
    const DDAS::HitExtension& extension(hit.getExtension());
    uint32_t crate = hit.GetCrateID();
    uint32_t slot  = hit.GetSlotID();
    uint32_t chan  = hit.GetChannelID();
    
    // Output the fit information:
    
    std::cout << "-------------------------------------\n";
    std::cout << "Crate " << crate << " slot " << slot << " chan " << chan <<std::endl;
    std::cout << "Single pulse fit\n";
    std::cout << " Chisquare: " << extension.onePulseFit.chiSquare << std::endl;
    std::cout << " A0 " << extension.onePulseFit.pulse.amplitude << std::endl;
    std::cout << " X0 " << extension.onePulseFit.pulse.position  << std::endl;
    std::cout << " K1 " << extension.onePulseFit.pulse.steepness << std::endl;
    std::cout << " K2 " << extension.onePulseFit.pulse.decayTime << std::endl;
    std::cout << " C  " << extension.onePulseFit.offset          << std::endl;
    
    std::cout << "\nDouble pulse fit\n";
    std::cout << "ChiSquare: " << extension.twoPulseFit.chiSquare << std::endl;

    bool plot = false;
    double x1 = extension.twoPulseFit.pulses[0].position;
    double x2 = extension.twoPulseFit.pulses[1].position;
    int left = 0, right=1;
    if (x1 > x2) {
      left = 1;
      right = 0;
    }
    int map[2] = {left, right};
    for (int p = 0; p < 2; p++) {
        std::cout << "A" << p << " " << extension.twoPulseFit.pulses[map[p]].amplitude << std::endl;
        std::cout << "X" << p << " " << extension.twoPulseFit.pulses[map[p]].position << std::endl;
        std::cout << "K" << p*2+1 << " " << extension.twoPulseFit.pulses[map[p]].steepness << std::endl;
        std::cout << "K" << p*2+2 << " " << extension.twoPulseFit.pulses[map[p]].decayTime << std::endl;

    }
    double ch1 = extension.onePulseFit.chiSquare;
    double ch2 = extension.twoPulseFit.chiSquare;

    double r = ch1/ch2;
    
    double dt = abs(x1 - x2);
    double A = extension.twoPulseFit.pulses[right].amplitude;
    if ((dt > 10) && (A > 900.0) && (A < 1200.0) && (r > 3.0)) plot = true;

    //    plot = true;

    if(plot)    plotFit(trace, extension);
    
    
}
/**
 * processItem
 *  Process a ring item:
 *  -  Break it down into fragments.
 *  -  Decode the ring item for each fragment using FitHitUnpacker.
 *  -  If a fragment has a fit; output the fit information and plot the
 *     trace and the two fits.
 * @param pItem - ring item.
 */
static void
processItem(CRingItem* pItem)
{
    uint16_t* pData = reinterpret_cast<uint16_t*>(pItem->getBodyPointer());
    FragmentIndex frags(pData);
    DAQ::DDAS::FitHitUnpacker unpacker;
    for (auto p = frags.begin(); p != frags.end(); p++) {
        DAQ::DDAS::DDASFitHit hit;
        unpacker.decode(p->s_itemhdr, hit);
        
        if (hit.hasExtension()) {
            reportHitFits(hit);
        }
    }
    
}


/**
 * main
 *    The entry point
 *    - Determine if the parameters are ok.
 *    - Open the data source.
 *    - Process ring items from that source.
 *
 * @param argc - number of command line parameters.
 * @param argv - The command line parameters.
 * @return int hopefully EXIT_SUCCESS
 */
int
main(int argc,char** argv)
{
    if (argc != 2) {
        usage(std::cerr, "Incorrect number of command line parameters");
    }
    
    const char* file = argv[1];
    
    try {
        std::vector<uint16_t> exclude;
        URL fileUrl = filenameToUri(file);
        CFileDataSource source(fileUrl, exclude);
        
        CRingItem* pItem(0);
        while(pItem = source.getItem())  {
            processItem(pItem);
            delete pItem;
        }
    }
    catch (CException& e) {
        std::cerr << "Exception caught" << e.ReasonText() <<std::endl;
        exit(EXIT_FAILURE);
    }
    catch (std::exception& e) {
        std::cerr << "Exception caught" << e.what() <<std::endl;
        exit(EXIT_FAILURE);
    }
    catch (std::string msg) {
        std::cerr << "Exception caught" << msg <<std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Unanticipated exception type caught\n";
        throw;                  // Maybe we'll figure out what to add.
    }
    return EXIT_SUCCESS;
}
