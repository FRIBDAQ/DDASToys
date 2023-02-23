/** 
 * @file  DDASDecoder.h
 * @brief Define an event proessor for reading ring items from a source, in 
 * this case a file. Very similar to the CEventProcessor class but really 
 * designed to work interactively with the GUI.
 */

/** @addtogroup traceview
 * @{
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

class DDASRingItemProcessor;

/**
 * @class DDASDecoder
 * @brief An interactive event processor integrated in the traceview GUI.
 *
 * An event processor class broken into parts performing specific actions 
 * like creating a data source or getting the next PHYSICS_EVENT. These 
 * functions can be hooked into the signal and slot framework used by Qt. 
 * See latest $DAQROOT/share/recipes/process/process.cpp for a more general 
 * example.
 */

class DDASDecoder
{
  // Private data
private:
  URL* m_pSourceURL;
  CDataSource* m_pSource;
  DDASRingItemProcessor* m_pProcessor;
  int m_count;
  
  // Canonicals
  
public:
  DDASDecoder();
  ~DDASDecoder();

  // Public methods
  
public:
  void createDataSource(std::string src);
  std::vector<DAQ::DDAS::DDASFitHit> getEvent();
  int skip(int nevts);

  /**
   * @brief Return the zero-indexed PHYSICS_EVENT number.
   * @return int  The event number.
   */
  int getEventCount() {return m_count-1;};
  std::string getFilePath(); // See .cpp since URL is forward declared

  // Private methods
  
private:
  CRingItem* getNextPhysicsEvent();
  void processRingItem(CRingItem& item);
};

#endif

/** @} */
