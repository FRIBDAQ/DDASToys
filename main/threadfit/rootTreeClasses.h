#include <TObject.h>
#include "RootExtensions.h"


class doubleFit : public TObject
{
public:
  UInt_t iterations;
  UInt_t fitStatus;
  Double_t chiSquare;
  RootPulseDescription pulse1;
  RootPulseDescription pulse2;
  Double_t offset;
  
  doubleFit(){}
  doubleFit(RootFit2Info& fit) {
    iterations = fit.iterations;
    fitStatus = fit.fitStatus;
    chiSquare = fit.chiSquare;
    pulse1    = fit.pulses[0];
    pulse2    = fit.pulses[1];
    offset    = fit.offset;
  }
  ClassDef(doubleFit, 1);

};
class fit : public TObject
{
public:
  std::vector<Bool_t>   haveExtension;
  std::vector<RootFit1Info> singleFits;
  std::vector<doubleFit> doubleFits;
  fit() {}
  RootHitExtension GetEntry(Int_t i) {
    RootHitExtension result;
    result.haveExtension = haveExtension[i];
    result.onePulseFit   = singleFits[i];
    result.twoPulseFit.chiSquare   = doubleFits[i].chiSquare;
    result.twoPulseFit.fitStatus   = doubleFits[i].fitStatus;
    result.twoPulseFit.pulses[0]   = doubleFits[i].pulse1;
    result.twoPulseFit.pulses[1]   = doubleFits[i].pulse2;
    result.twoPulseFit.offset      = doubleFits[i].offset;    
    return result;
  }
  RootHitExtension operator[](Int_t i) {
    return GetEntry(i);
  }
  void addFit(RootHitExtension& hit) {
    haveExtension.push_back(hit.haveExtension);
    singleFits.push_back(hit.onePulseFit);
    doubleFits.push_back(doubleFit(hit.twoPulseFit));
  }
  void clear() {
    singleFits.clear();
    doubleFits.clear();
    haveExtension.clear();
  }
  ClassDef(fit, 1);
};

ClassImp(doubleFit);
ClassImp(fit);
