/** @file:  FitEditorTemplate.h
 *  @brief: FitEditor class for analytic fitting
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
 *   Extend the hit with the template fitting information, overwriting any 
 *   existing extension. It's intended for use with the EventEditor framework 
 *   providing a complete description of the new event body.
 */

class FitEditorTemplate : public CBuiltRingItemEditor::BodyEditor
{
public:
  FitEditorTemplate();
  virtual ~FitEditorTemplate();

  // Mandatory interface from CBuiltRingItemEditor::BodyEditor
public:
  virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody);
  virtual void free(iovec& e);

  // Additional functionality for this class
private:
  int pulseCount(DAQ::DDAS::DDASHit& hit);
  
  // Private member data
private:
  Configuration* m_pConfig;
};

#endif
