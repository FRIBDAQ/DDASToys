/**
 * @file  eeconverter.cpp
 * @brief Create an event processor, call its operator() to process data, 
 * and handle exceptions.
 */ 

#include <string>
#include <iostream>

#include <Exception.h>

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
  catch(CException& e) {
    std::cerr << "eeconverter main caught an exception: "
	      << e.ReasonText() << std::endl;
    return EXIT_FAILURE;
  }
  catch(std::invalid_argument& e) {
    std::cerr << "eeconverter main caught an exception: "
	      << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch (std::string msg) {
    std::cerr << "eeconverter main caught an exception: "
	      << msg << std::endl;
    return EXIT_FAILURE;
  }
  catch (...) {
    std::cerr << "eeconverter main caught an unexpected exception type\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
