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

/** @file:  pulsmaker.cpp
 *  @brief: Make a bunch of pulses  with random positions, amplitudes baselines and
 *          noise.
 */
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include "functions.h"

/**
 *  Usage:
 *      pulsemaker file npulses noise-amplitude 1 | 2 [saturation]
 *   Where:
 *       file     - the name of the output file to write.
 *                  Will be overwritten if exists
 *       npulses  - Number of pulses to write.
 *       noise-amplitude - amplitude of the noise.  This is uniformly distributed
 *                  and clamped so that the trace is never negative.
 *       1 | 2    -  Determines if the traces have one or two pulses.
 *       saturation - optional
 *
 *  The output files contain  a header line that contains the numbre of traces
 *  in the file.
 *
 *  This is followed by pulses that contain a one line header.
 *  The header contains nsamples A1 K1 K2 X1 C A2 K3 K4 X2
 *
 *  If there is only one pulse, parameters for the second pulse are all zero.
 *  nsamples is the number of samples in a trace.
 */

/**
 *  The ranges for all but noise are fixed values:
 */

const unsigned SAMPLES(500);        // Samples in a trace.

const unsigned A_LOW(10);            // Smallest allowed amplitude.
const unsigned A_HIGH(8192);        // '14 bit adc'.

const unsigned BASELINE_LOW(10);
const unsigned BASELINE_HIGH(100);

const double K1_LOW(0.1);          // Shallowest rise.
const double K1_HIGH(0.9);         // steepest rise.

const double K2_LOW(0.00005);           // Shallowest decay.
const double K2_HIGH(0.00007);          // Steepest decay.

// These interact with SAMPLES -- if pulses are to be fully in the trace.

const unsigned X_LOW(100);             // earliest pulse1 onset.
const unsigned X1_HIGH(250);            // latest pulse1  onset.
const unsigned X2_HIGH(400);            // latest pulse 2 onset (earliest is X1 position).

unsigned saturation(0xffff);      // If pulse saturation level not set.
/**
 * rrange
 *    Random numnber in the specified range.
 * @param low - lo limit of the number.
 * @param high - Upper limit.
 */
static double
rrange(double low, double high)
{
    double range = high-low;
    double result =  range*drand48() + low;
    
    if ((result < low) || (result > high)) {
        std::cerr << "random failed range: " << low << " : " << high
            << " got " << result;
    }
    return result;
}

/**
 * generateSinglePulses
 *    Makes a file containing singlepulses.
 *
 * @param o  - Stream to which the file is written.
 * @param n  - Number of pulses to write.
 * @param a  - noise amplitude.
 */
static
void generateSinglePulses(std::ostream& o, int n, int a)
{
    for (int i = 0; i < n; i++)  {
        double A = rrange(A_LOW, A_HIGH);
        double C = rrange(BASELINE_LOW, BASELINE_HIGH);
        double K1 = rrange(K1_LOW, K1_HIGH);
        double K2= rrange(K2_LOW, K2_HIGH);
        double X1 = rrange(X_LOW, X1_HIGH);
        
        // Header line
        
        o << SAMPLES << " " << A << " " << K1 << " " << K2 << " " << X1 << " " << C
            << " 0 0 0 0 \n";
        // Trace:
        
        for (int t = 0; t < SAMPLES; t++) {
            double p = DDAS::singlePulse(A, K1, K2, X1, C, t) + rrange(-a, a);
            if (p < 0.0) p = 0.0;           // Clamp the trace.
            if (p > saturation) p = saturation;
            o << t << " " << p << std::endl;
        }
    }
}


/**
 * generateDoublePulses
 *    Generate random doublepulse file.
 *
 * @param o   - output stream.
 * @param n   - Number of traces to make.
 * @param a   - Noise Amplitude.
 */
static
void generateDoublePulses(std::ostream& o, int n, int a)
{
   for (int i = 0; i < n; i++)  {
        double A1 = rrange(A_LOW, A_HIGH);
        double C = rrange(BASELINE_LOW, BASELINE_HIGH);
        double K1 = rrange(K1_LOW, K1_HIGH);
        double K2= rrange(K2_LOW, K2_HIGH);
        double X1 = rrange(X_LOW, X1_HIGH);
        
        double A2 = rrange(A_LOW, A_HIGH);
#if COMMON_TIMING==1
        double K3 = K1;       // Same time parameters for both pulses.
        double K4 = K2;
#else
        double K3 = rrange(K1_LOW, K1_HIGH);
        double K4 = rrange(K2_LOW, K2_HIGH);
#endif
        double X2 = rrange(X1, X2_HIGH);
        
        // Header line
        
        o << SAMPLES << " " << A1 << " " << K1 << " " << K2 << " " << X1 << " " << C
            <<  " " << A2 << " " << K3 << " " << K4 << " " << X2 << std::endl;
        // Trace:
        
        for (int t = 0; t < SAMPLES; t++) {
            double p = 
                DDAS::doublePulse(A1, K1, K2, X1, A2, K3, K4, X2, C, t) +
                    rrange(-a, a);
            if (p < 0.0) p = 0.0;           // Clamp the trace.
            if (p > saturation) p = saturation;
            o << t << " " << p << std::endl;
        }
    }    
}
/**
 * iConvert
 *    Convert a C string to an unsigned integer value.
 *
 * @param str - string to convert.
 * @return int - The value.
 * @retval -1  - Conversion failed or < 0.
 */
static int
iConvert(const char* str)
{
    char* end;
    
    int result = strtol(str, &end, 0);
    if (end == str) return -1;
    
    return result;
}

/**
 * usage
 *    Complain about misuse and abuse of the program parameters.
 *
 * @param o - output stream where messages are written
 * @param msg     - Error message to precedethe usage.
 */
static void
usage(std::ostream& o, const char* msg)
{
    o << msg << std::endl << std::endl;
    o << "Usage:\n";
    o << "     pulsemaker file npulses noise-amplitude 1|2 [saturation]\n";
    o << "Where:\n";
    o << "   file     - Name of the file to write\n";
    o << "   npulses  - Number of pulses to write\n";
    o << "   noise-amplitude - Amplitude of the noise put on the pulses.\n";
    o << "   1|2       - 1 to write single pulses 2 writes double pulses\n";
    o << "   saturation - optional value at which the pulse saturates";
    
    exit(EXIT_FAILURE);       // Writing this message means we gave up.
}
/**
 * main
 *     Entry point - see comments and usage function for command usage.
 *  @param argc  - number of command line parameters
 *  @param argv  - array of pointers to the parameter strings (includes program).
 */
int main(int argc, char** argv)
{
    // Need exactly 5 parameters.
    
    if (argc != 5 && argc != 6) {
        usage(std::cerr, "Incorrect number of command line parameters.");
    }
    
    const char* file       = argv[1];
    const char* strNpulses = argv[2];
    const char* strNoise   = argv[3];
    const char* strSelect  = argv[4];          // 1|2.
    
    // Convert the integer parameters.   Cry foul if they can't be converted to
    // +-ive integers.
    
    int nPulses = iConvert(strNpulses);
    if (nPulses <= 0) {
        usage(std::cerr, "Number of pulses must be an integer greater than zero");
    }
    int noise = iConvert(strNoise);
    if (noise < 0) {
        usage(std::cerr, "Noise amplitude must be an integer >= 0");
    }
    int pulses;
    if (strcmp(strSelect, "1") == 0) {
        pulses = 1;
    } else if (strcmp(strSelect, "2") == 0) {
        pulses = 2;
    } else {
        usage(std::cerr, "Can only select 1 or 2 pulses");
    }
    
    if (argc == 6) {          // Saturation supplied.
        int userSat = iConvert(argv[5]);
        if (userSat <= 0) {
            usage(std::cerr, "Saturation value must be an integer > 0");
           
        }
         saturation = userSat;
    }
    
    // open the output file; seed the random generator.
    
    std::ofstream o(file);
    srand48(time(NULL));            // Only better would be a long from/ dev/random
    
    // Generate the values:
    
    if (pulses == 1) {
        generateSinglePulses(o,  nPulses, noise);
    } else {
        generateDoublePulses(o, nPulses, noise);
    }
    return EXIT_SUCCESS;
}



