
// implement ddas analysis.

#include "CDDASAnalyzer.h"
#include "Outputter.h"

#include <FragmentIndex.h>
#include <CRingItemFactory.h>
#include <DataFormat.h>
#include <CRingItem.h>
#include <vector>

#include <DDASHitUnpacker.h>
#include <DDASHit.h>
#include <DataFormat.h>
#include <string.h>
#include "lmfit.h"

using namespace DAQ::DDAS;

/**
 * constructor
 *
 * @param id - Identifies us.
 * @param out - Object we'll use to output data.
 */
CDDASAnalyzer::CDDASAnalyzer(int id, Outputter& out) :
  m_nId(id), m_Outputter(out), m_singlePredicate(0), m_doublePredicate(0)
{
  
}

void
CDDASAnalyzer::operator()(const void* Data)
{
  const CRingItem* p = reinterpret_cast<const CRingItem*>(Data);
  if (p->type() == PHYSICS_EVENT) {
      CRingItem* pItem = CRingItemFactory::createRingItem(*p);
      processPhysicsItem(pItem);
			delete pItem;
  }
  
}
/**
 * end
 *    Called when there's no more data to process.
 */
void
CDDASAnalyzer::end()
{
  m_Outputter.end(m_nId);
}

void CDDASAnalyzer::setSingleFitPredicate(CDDASAnalyzer::FitPredicate* p) {
  m_singlePredicate = p;
}
void CDDASAnalyzer::setDoubleFitPredicate(CDDASAnalyzer::FitPredicate* p) {
  m_doublePredicate = p;
}

/**
 processPhysicsItem
   This does the actual analysis:
   We're going to build up a vector of output fragments, for an output ring item.
   - Each fragment is assumed to be  DDASHit.
   - If a fragment has no waveform, it's output unmodified.
   - If a fragment has a waveform, the waveform is fit with 
   - first a single step function and then a double step function.
   - Additional data (all doubles - except the size) is then appended to the fragment:
      - Number of bytes in the appendix.
      - chisquare of single pulse fit.
      - position of edge of single pulse fit.  (x1)
      - amplitude of single pulse fit.         (A1)
      - rise time parameter of single pulse fit. (k1)
      - Decay rate of single pulse fit.        (k3)
      - constant offset of single pulse fit    (C)
      - chisquare of double pulse fit
      - position of first edge                  x1
      - amplitude of first pulse                A1
      - rise time parmeter of first edge        k1
      - decay time parameter of first pulse     k3
      - position of second edge                 x2
      - amplitude of second pulse               A2
      - rise time parameter of second edge      k2
      - decay time parameter of second pulse    k4
      - constant parameter of double plse fit   C

@param pItem -pointer to the undifferentiated ring item.

*/
void
CDDASAnalyzer::processPhysicsItem(CRingItem* pItem)
{
  FragmentIndex frags(reinterpret_cast<uint16_t*>(pItem->getBodyPointer()));
  size_t nFrags = frags.getNumberFragments();
  std::vector<FragmentInfo> outputFrags;
  DDASHitUnpacker unpacker;
  
  for (int i =0; i < nFrags; i++) {
    FragmentInfo f = frags.getFragment(i);
    DDASHit hit;
    uint32_t* begin = reinterpret_cast<uint32_t*>(f.s_itembody);
    uint32_t* end   = reinterpret_cast<uint32_t*>(
	  reinterpret_cast<uint8_t*>(f.s_itembody) + f.s_size
    );
    unpacker.unpack(begin, end, hit);
    if (hit.GetTraceLength()) {
      // Must fit...

      // Add the modified fragment to the event.  Note that this fragment
      // has had
      
      fit (f, hit, hit.GetTrace());
      outputFrags.push_back(f);
    } else {
      // Unchanged:

      outputFrags.push_back(f);
    }
    // Send the potentially modified fragment vector on to the farmer.
    // The farmer will write it out and free storage allocated for the fit
    // fragments.
    
    
  }
  outData out =
      {pItem->getEventTimestamp(), &outputFrags};
  m_Outputter.outputItem(m_nId, &out);
}
/**
 *   fit
 *      Fit the trace in the fragment:
 *      - Space is allocated for the new fragment body (old fragment)
 *      - The new fragment body is plugged into the fragment.
 *      - The fits are computed into their slots in the new fragment body.
 *
 *  @param[inout] frag - Reference to the fragment info struct that describes this
 *                fragment
 *  @param[in]    hit - references the DDAS hit information.
 *  @param[in]    trace - The trace points from the hit.
 */
void
CDDASAnalyzer::fit(
  FragmentInfo& frag, DAQ::DDAS::DDASHit& hit, std::vector<uint16_t>& trace
)
{
  //
  // If neither predicate says we should do a fit just return right away:
  //
  std::pair<std::pair<unsigned, unsigned>, unsigned> fit1Info =
    { {0, trace.size() - 1}, 0xffff};
  std::pair<std::pair<unsigned, unsigned>, unsigned> fit2Info(fit1Info);
  
  // If there is a predicate for the first fit, check it.
  // Don't do anything if it returns limits that are equal.
  
  if (m_singlePredicate) {
    fit1Info = (*m_singlePredicate)(frag, hit, trace);  
  }
  std::pair<unsigned, unsigned> fit1Limits     = fit1Info.first;
  unsigned                      fit1Saturation = fit1Info.second;

  // fit limits that have no extent indicate don't do the fit.  If
  // fit1 should not be done, neither should fit 2. so we're done here.
  
  if (fit1Limits.first == fit1Limits.second) return;
  
  
  // Allocate the new fragment.  We're going to assume that the fragment
  // has a body  header.

  size_t newSize = frag.s_size + sizeof(DDAS::HitExtension);
  size_t originalSize = frag.s_size;
  
  void* newItem = calloc(1, newSize);
  
  // Need to update the ring item header for the payload too:
  
  pRingItemHeader pHeader =
    reinterpret_cast<pRingItemHeader>(frag.s_itemhdr);
  pHeader->s_size += sizeof(DDAS::HitExtension);
  
  memcpy(newItem, frag.s_itemhdr, originalSize);     // Copy the old fragment.
  frag.s_itemhdr = reinterpret_cast<uint16_t*>(newItem);
  frag.s_itembody = findBody(newItem);
  
  // Update the fragment size and get a pointer to the event extension:
  
  
  frag.s_size = newSize;
  
  // The HitExtension will be put right at the end of the copy of the original data.
  
  uint8_t* pItem = reinterpret_cast<uint8_t*>(newItem);
  DDAS::HitExtension* pExtension =
      reinterpret_cast<DDAS::HitExtension*>(pItem + originalSize);
      
  
  DDAS::lmfit1(&(pExtension->onePulseFit), trace, fit1Limits, fit1Saturation);
  
  // See what the limits of the second fit should be... or if it even
  // should be done.  Note that we can feed the result of lmfit1
  //  back into lmfit2 as an attempt to guess the answer
  
  if (m_doublePredicate) {
    fit2Info = (*m_doublePredicate)(frag, hit, trace);
  }
  std::pair<unsigned, unsigned> fit2Limits     = fit2Info.first;
  unsigned                      fit2Saturation = fit2Info.second;
  if (fit2Limits.first == fit2Limits.second) return;
  
  DDAS::lmfit2(&(pExtension->twoPulseFit), trace, fit2Limits,
               &(pExtension->onePulseFit), fit2Saturation);
  
}


/**
 * findBody
 *    Find the body of a ring item.
 * @param pItem - pointer to the ring item.
 * @return uint16_t*  pointer to the body (past body header) of the item.
 */
uint16_t*
CDDASAnalyzer::findBody(void* pItem)
{
  uint8_t* pRingItem8 = reinterpret_cast<uint8_t*>(pItem);
  uint8_t* pBodyHeader= pRingItem8 + sizeof(RingItemHeader);
  
  // If pBodyHeader points to a uint32_t 0 then the body just follows that
  // (my idiot design mistake if I'd only used a value of sizeof(uint32_t) this would
  // have been fully mechanical).
  
  uint32_t *pBodyHeaderSize = reinterpret_cast<uint32_t*>(pBodyHeader);
  uint32_t bodyHeaderSize = *pBodyHeaderSize;
  if (bodyHeaderSize == 0) {
    return reinterpret_cast<uint16_t*>(pBodyHeaderSize+1); // skipping just count.
  } else {
    pBodyHeader += bodyHeaderSize;      // Skip the full body header.
    return reinterpret_cast<uint16_t*>(pBodyHeader);
  }
}