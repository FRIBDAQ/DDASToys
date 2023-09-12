/**
 * @file  eeconverter.cpp
 * @brief Create an event processor, call its operator() to process data, 
 * and handle exceptions.
 */ 

#include <string>
#include <iostream>

#include "CEventProcessor.h"

//____________________________________________________________________________
/**
 * @brief eeconverter main.
 *
 * @param argc Number of command line arguments.
 * @param argv Array containing the command line arguments.
 *
 * @return int
 * @retval EXIT_SUCCESS If the program executes and completes properly.
 * @retval EXIT_FAILURE Otherwise.
 *
 * @details
 * Creates and calls the event processor's operator(). Handles exceptions 
 * thrown by other parts of the program (hopefully).
 */
int main(int argc, char** argv)
{  
    CEventProcessor ep;
    try {
	ep(argc, argv);
    }
    catch (...) {
	std::cerr << "ERROR: eeconverter main caught an unexpected exception\n";
	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
