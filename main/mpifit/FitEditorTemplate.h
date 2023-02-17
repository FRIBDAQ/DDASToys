/** 
 * @file  FitEditorTemplate.h
 * @brief Definition of the FitEditor class for template fitting.
 */

#ifndef FITEDITORTEMPLATE_H
#define FITEDITORTEMPLATE_H

#include <CBuiltRingItemEditor.h>

#include <vector>

namespace DAQ {
  namespace DDAS {
    class DDASHit;
  }
}

class Configuration;

/**
 * @class FitEditorTemplate
 * @brief Fit trace data with the template fitting functions and add hit 
 * extensions.
 *
 * Editing the hit overwrites any existing extension. This class is intended 
 * for use with the EventEditor framework providing a complete description of 
 * the new event body.
 */

class FitEditorTemplate : public CBuiltRingItemEditor::BodyEditor
{
public:
  FitEditorTemplate();
  virtual ~FitEditorTemplate();

  // Mandatory interface from CBuiltRingItemEditor::BodyEditor
public:
  virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize, void* pBody);
  virtual void free(iovec& e);

  // Additional functionality for this class
private:
  int pulseCount(DAQ::DDAS::DDASHit& hit);
  
  // Private member data
private:
  Configuration* m_pConfig; //! Configuration file parser.
};

#endif
