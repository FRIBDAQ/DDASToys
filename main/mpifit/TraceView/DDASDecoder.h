/** @file: DDASDecoder.h
 *  @brief: Define an event proessor for reading ring items from a source. 
 *  Generally a file for this application. Very similar to the CEventProcessor 
 *  class but really designed to work interactively with the plotter.
 */

#ifndef DDASDECODER_H
#define DDASDECODER_H

#include <string>
#include <vector>

class URL;
class CRingItem;
class CDataSource;
namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}

class DDASEventProcessor;

class DDASDecoder
{
  // Private data
private:
  URL* m_pSourceURL;
  CDataSource* m_pSource;
  DDASEventProcessor* m_pProcessor;

  // Canonicals
public:
  DDASDecoder();
  ~DDASDecoder();

  // Public methods
public:
  void createDataSource(std::string src);
  std::vector<DAQ::DDAS::DDASFitHit> getEvent();

  // Private methods
private:
  void processRingItem(CRingItem& item);
};

#endif
