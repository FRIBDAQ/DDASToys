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

/** @file:  fit1.cpp
 *  @brief: Fits single pulses to  pulses in a pulsemaker file.
 */

#include "functions.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <stdint.h>

/**
 * Usage:
 *    fit1  pulsemaker-file
 *    
 *  pulsemaker-file - is the filename of a file written by pulsemaker.
 */

/**
 * outputTrace
 *     output the trace information:
 *    Actual function parameters,
 *    Fit function parameters
 *    Plot of trace, fit, and differences.
 *
 *  @param result - fit results from lmfit1
 *  @param trace  - trace points.
 *  @param A      - Actual amplitude.
 *  @param k1     - Actual rise parameter.
 *  @param k2     - Actual decay parameter.
 *  @param x1     - Actual pulse position.
 *  @param C      - Actual DC Offset.
 */
static
void outputTraceInfo(
    const DDAS::fit1Info& result, const std::vector<uint16_t>& trace,
    double A, double k1, double k2, double x1, double C
)
{
    std::cout << "--------------------------------\n";
    std::cout << "Chisquare " << result.chiSquare << std::endl;
    std::cout << "A  " << result.pulse.amplitude << " : " << A << std::endl;
    std::cout << "K1 " << result.pulse.steepness << " : " << k1 << std::endl;
    std::cout << "K2 " << result.pulse.decayTime << " : " << k2 << std::endl;
    std::cout << "X1 " << result.pulse.position  << " : " << x1 << std::endl;
    std::cout << "C  " << result.offset         << " : " << C  << std::endl;
    
    // Write a diffplot file with the x, trace, fit, and difference.

    {    
        std::ofstream o("fit1data.dat");
        for (unsigned i =0; i < trace.size(); i++) {
            double y = DDAS::singlePulse(
                result.pulse.amplitude, result.pulse.steepness, result.pulse.decayTime,
                result.pulse.position, result.offset, i
            );
            double diff = y - trace[i];
            o << i << " " << y << " " << trace[i] << " "
              << diff*diff/trace[i] << std::endl;
        }
    }
    // use system to run diffplot on the file we just created.
    
    system("diffplot fit1data.dat");
}

/**
 * fitNext
 *   Fit the next trace from the file.  Note that this does nothing if the next
 *   read of the title line failed as that's probably  just the null line
 *   at the end of the file.
 * @param in - input stream from which to read the data.
 */
static void
fitNext(std::istream& in)
{
    int npts;
    double A;
    double k1;
    double k2;
    double x1;
    double C;
    double junk;            // For unused second pulse info.
    
    // Read the title line:
    
    in >> npts >> A >> k1 >> k2 >> x1 >> C >> junk >> junk >> junk >>junk;
    if (in.fail()) return;
    
    // truncate the trace into integers:
    bool ok(true);
    std::vector<uint16_t> trace;
    for (int i =0; i < npts; i++) {
        int x;
        double y;
        
        in >> x >> y;
        if (y > UINT16_MAX) {
            std::cout << "Trace has bad points (> " << UINT16_MAX << ")\n";
            std::cout << "abandoning\n";
            ok = false;
        }
        trace.push_back(static_cast<uint16_t>(y));
    }
    std::pair<unsigned, unsigned> limits(0, trace.size() -1);
    if (ok) {
        //
        
        DDAS::fit1Info result;
        DDAS::lmfit1(&result, trace, limits);
        outputTraceInfo(result, trace, A, k1, k2, x1, C);
    }
}

/**
 *  usage:
 *     Output error message and program usage.  exits.
 * @param o  - stream to which the output is done.
 * @param msg - Error message.
 */
static void
usage(std::ostream& o, const char* msg)
{
    o << msg << std::endl << std::endl;
    o << "Usage\n";
    o << "    fit1 pulsemaker-file\n";
    o << "Where:\n";
    o << "   pulsemaker-file - is the name of a file written by pulsemaker\n";
    
    exit(EXIT_FAILURE);
}


/**
 * main
 *    Open the input file and fit/plot for each trace in the file.
 *    Note that a file is written for each fit named
 *    fit1data.dat that is in the format needed by diffplot -- which is run
 *    via system to display the fit and differences.
 *
 * @param argc   - parameter count.
 * @param argv   - Pointers to parameters
 * @return int (hopefully EXIT_SUCCESS).
 *
 */
int
main(int argc, char** argv)
{
    if (argc !=2 ) {
        usage(std::cerr, "Incorrect number of command line parameters");
    }
    const char* filename = argv[1];
    
    std::ifstream rawData(filename);
    if (rawData.fail()) {
        usage(std::cerr, "Could not open data filename\n");
    }
    
    while(!rawData.eof()) {
        fitNext(rawData);
    }
    
    return EXIT_SUCCESS;
}