// class to analyze ddas data.

#ifndef CDDASANALYZER_H
#define CDDASANALYZER_H
#include <stdint.h>
#include "functions.h"
#include <DDASHit.h>


/**
 *  @todo Hoist this as a subclass for an abstract base class analyzer.
 */

class CRingItem;
class Outputter;
struct FragmentInfo;

class CDDASAnalyzer {
public:
    struct outData {
        uint64_t timestamp;
        void* payload;
    };
    // Abstract Base classe for various predicates:
    
    class FitPredicate {
    public:
        FitPredicate()  {}
        virtual ~FitPredicate() {}
        virtual std::pair<std::pair<unsigned, unsigned>, unsigned> operator()(
            const FragmentInfo& frag, DAQ::DDAS::DDASHit& hit,
            const std::vector<uint16_t>& trace
        ) =0;
    };
private:
    int m_nId;
    Outputter& m_Outputter;
    FitPredicate* m_singlePredicate;
    FitPredicate* m_doublePredicate;
public:
    CDDASAnalyzer(int id, Outputter& out);
    void operator()(const void* pData);
    void end();
    void setSingleFitPredicate(FitPredicate* p);
    void setDoubleFitPredicate(FitPredicate* p);
private:
    void processPhysicsItem(CRingItem* pItem);
    void fit(
        FragmentInfo& frag, DAQ::DDAS::DDASHit& hit,
        std::vector<uint16_t>& trace
     );
    
    uint16_t* findBody(void* pItem);
  
};

#endif
