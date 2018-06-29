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

/** @file:  functiontests.cpp
 *  @brief: Do some testing of the functions.h/cpp code.
 *
 */

#include "functions.h"
#include "lmfit.h"
#include <iostream>
#include <fstream>
void
static out(std::ostream& o, double x, double y)
{
    o << x << " " << y << std::endl;
}


int main(int argc, char** argv)
{
    // write a file containin logistic function points from 0-500
    // the mid point is at 250, the amplitude is 100, and the steepness is
    // 1.0
    
    {
        std::ofstream o("logistic.dat");
        o << "Logistic 100, 0.5, 250\n";
        for (int i =0; i < 500; i++) {
            double x = i;
            double value = DDAS::logistic(100.0, 0.5, 250.0, x);
            out(o, x, value);
            
        }
    }                      // closes the file
    
    // Write a decay curve:
    //  A = 100, k = 1/50.0, x1 = 250.
    {
        std::ofstream o("decay.dat");
        o << "Decay 100 1/50, 250\n";
        for (int i = 0; i < 500; i++) {
            double x = i;
            double value = DDAS::decay(100.0, 1/50.0, 250.0, x);
            out(o, x, value);
        }
    }
    // Write a switch on curve.  Switches on at 250.
    
    {
        std::ofstream o("switchon.dat");
        o << "Step function at 250\n";
        for (int i = 0; i < 500; i ++) {
            double x = i;
            double value = DDAS::switchOn(250.0, x);
            out(o, x, value);
        }
    }
    double A1= 10;
    double k1= 0.1;
    double k2 = 1/50.0;
    double x1 = 150.0;
    double C  = 50.0;
    
    double A2 = 100;
    double k3 = 0.25;
    double k4 = 1/25.0;
    double x2 = 300.0;
    // Write a single pulse
    // Amplitude, 100, rise factor 0.5, decay time 1/50, position 250
    // dc offset 50
    
    {
        std::ofstream o("singlepulse.dat");
        o << "Single pulse a =100 rise 0.5, decay 1/50, x1 = 250 dc 50\n";
        for (int i =0; i < 500; i++) {
            double x = i;
            double y = DDAS::singlePulse(A1, k1, k2, x1, C, x);
            out(o, x, y);
        }
    }
    // Write a double pulse
    //  A1 = 100,  K1 = 0.5 K2 = 1/50 X1 = 150.0 DC = 50-
    //  A2 = 50,   k3 = 0.25 k4 = 1/25 X2 = 200
    
    {
        std::ofstream o("doublepulse.dat");
        o << "Double pulse A1=100 k1=0.5 k2=1/50 x1 = 150 C = 50 A2=50 k3 =0.25 k4 =1/25 x2=200\n";
        for (int i =0; i < 500; i++) {
            double x = i;
            double y = DDAS::doublePulse(
                A1, k1, k2, x1,
                A2, k3, k4, x2,
                C, x
                                         );
            out(o, x, y);
        }
    }
    // If I make a  pulse trace with the specified set of parameters,
    // the chisquare with those parameters should be really close to zero
    // (not exactly since traces are truncated integers)
    
    std::vector<uint16_t> pulse1;

    
    for (int i = 0; i < 500; i++) {
        double x = i;
        pulse1.push_back(DDAS::singlePulse(A1, k1, k2, x1, C, x));
    }
    std::cout << "Pulse 1 chisquare is: "
        << DDAS::chiSquare1(A1, k1, k2, x1, C, pulse1)
        << std::endl;
    
    // Same thing for double pulse:
    

    
    std::vector<uint16_t> pulse2;
    for (int i =0; i < 500; i++) {
        double x = i;
        pulse2.push_back(DDAS::doublePulse(A1, k1, k2, x1, A2, k3,k4, x2, C, x));
    }
    std::cout << "Pulse 2 chi square is : "
        << DDAS::chiSquare2(A1, k1, k2, x1, A2, k3, k4, x2, C, pulse2)
        << std::endl;
        
    // Try to fit the single pulse trace.
    std::pair<unsigned, unsigned> limits(0, pulse1.size() -1);  
    DDAS:: fit1Info result1;
    DDAS::lmfit1(&result1, pulse1, limits);
    std::cout << "--------------  Fit pulse 1 -------------\n"
        << "Chi square:  " << result1.chiSquare << std::endl
        << "A1        :  " << result1.pulse.amplitude << " " << A1 << std::endl
        << "k1        :  " << result1.pulse.steepness << " " << k1 << std::endl
        << "k2        :  " << result1.pulse.decayTime << " " << k2 << std::endl
        << "x1        :  " << result1.pulse.position << " " << x1 << std::endl
        << "C         :  " << result1.offset <<  " " << C << std::endl;
        
        
    {
        std::ofstream o("fit1.dat");
        for (int i =0; i < pulse1.size(); i++) {
            double x = i;
            double yf = DDAS::singlePulse(
                result1.pulse.amplitude, result1.pulse.steepness,
                result1.pulse.decayTime, result1.pulse.position, result1.offset,
                x
            );
            double ya = pulse1[i];
            double diff = yf - ya;
            
            o << x << " " << yf << " " << ya << " " << diff*diff/yf << std::endl;
        }
    }
    // Fit 2 pulses.
    
    DDAS::fit2Info result2;
    DDAS::lmfit2(&result2, pulse2, limits);
    
    std::cout << "---------------------------- fit double pulse --------------\n";
    std::cout << "Chi square: " << result2.chiSquare << std::endl;
    
    std::cout << "A1   : " << result2.pulses[0].amplitude << " " << A1 << std::endl;
    std::cout << "k1   : " << result2.pulses[0].steepness << " " << k1 << std::endl;
    std::cout << "k2   : " << result2.pulses[0].decayTime << " " << k2 << std::endl;
    std::cout << "x1   : " << result2.pulses[0].position << " "  << x1 << std::endl;
    std::cout << std::endl;
    std::cout << "A2   : " << result2.pulses[1].amplitude << " " << A2 << std::endl;
    std::cout << "k3   : " << result2.pulses[1].steepness << " " << k3 << std::endl;
    std::cout << "k4   : " << result2.pulses[1].decayTime << " " << k4 << std::endl;
    std::cout << "x2   : " << result2.pulses[1].position << " "  << x2 << std::endl;
    
    std::cout << "C    : " << result2.offset << " " << C << std::endl;
    {
        std::ofstream o("fit2.dat");
        for (int i =0; i < pulse2.size(); i++) {
            double x = i;
            double pulse = pulse2[i];
            double fit   = DDAS::doublePulse(
                result2.pulses[0].amplitude, result2.pulses[0].steepness,
                result2.pulses[0].decayTime, result2.pulses[0].position,
                
                result2.pulses[1].amplitude, result2.pulses[1].steepness,
                result2.pulses[1].decayTime, result2.pulses[1].position,
                
                result2.offset, x
            );
            double diff(fit - pulse);
            o << x << " " << fit << " " << pulse << " " << diff*diff/pulse << std::endl;
        }
    }
}