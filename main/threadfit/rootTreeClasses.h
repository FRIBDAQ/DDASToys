class doubleFit {
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
  
};
class fit {
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
}; 

Int_t reserializer(){
  // input file
  TFile *fin = new TFile("/mnt/analysis/e16032/rootfiles/run-0182-00.root","UPDATE");
  TTree *treein = (TTree*)fin->Get("DDASFit");

  std::vector<RootHitExtension>* old_extension(0);
  treein->SetBranchAddress("HitFits", &old_extension);  

  // initialization class of vector
  fit* item = 0;
  item = new fit();
  
  // output file
  TFile *f = new TFile("./new-run-0182-00.root","RECREATE");
  TTree *tree = new TTree("DDASFit", "");
  tree->Branch("newHitFits", "fit", item);

  Int_t nentries = treein->GetEntries();
  cout << "I have " << nentries << " entries" << endl;
  for (int i = 0; i < nentries; ++i) {
    treein->GetEntry(i);
    if ((i % 10000) == 0)std::cout << i << std::endl;
    std::vector<RootHitExtension>& oldExt(*old_extension);
    Int_t size = oldExt.size();
    for (int h = 0; h < size; h++) {
      item->addFit(oldExt[h]);
    }
    tree->Fill();
    item->clear();
  }
  std::cout << "Reserializer done" << std::endl;
  tree->Write();
  std::cout << "File written" << std::endl;

  delete f;
  return 0;
}
