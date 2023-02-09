/** @file: QHitData.cpp
 *  @brief: Implementation of hit data management class.
 */

#include "QHitData.h"

#include <iostream>
#include <iomanip>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QPushButton>
#include <QGroupBox>

#include <DDASFitHit.h>
#include "FitManager.h"

//____________________________________________________________________________
/**
 * Constructor
 *   Initialize child widgets and define main layout.
 *
 * @param pFitMgr - pointer to FitManager object used by this class, managed 
 *                  by caller
 * @param parent - pointer to QWidget parent object, default = nullptr
 */
QHitData::QHitData(FitManager* pFitMgr, QWidget* parent) :
  QWidget(parent), m_pFitManager(pFitMgr), m_pExtension(new DDAS::HitExtension)
{
  QGroupBox* hitBox = createHitBox();
  QGroupBox* classifierBox = createClassifierBox();
  QGroupBox* fitBox = createFitBox();
  
  QHBoxLayout* mainLayout = new QHBoxLayout;
  mainLayout->addWidget(hitBox);
  mainLayout->addWidget(classifierBox);
  mainLayout->addWidget(fitBox);
  setLayout(mainLayout);

  // Once the box has been configured with the default fitting method, we use
  // the configuration reader to get the information we need to display the
  // fits by reading the associated environment variables.
  // Some issues:
  //   - Pointing to the wrong files exits with no chance to correct
  //   - No ability to reconfigure to different configuration files on the
  //     fly (though why would you want to?)
  // Then create the connections to handle switches between fitting methods.
  
  configureFit();
  createConnections();
}

//____________________________________________________________________________
/**
 * Destructor
 */
QHitData::~QHitData()
{}

//____________________________________________________________________________
/**
 * update
 *   Update hit data and enable printing of fit information to stdout if the 
 *   hit has an extension.
 *
 * @param hit - references the hit we are processing
 */ 
void
QHitData::update(const DAQ::DDAS::DDASFitHit& hit)
{
  updateHitData(hit);
  if (hit.hasExtension()) {
    m_pPrintFit->setEnabled(true);

    // Calling a slot function with an non-standard type argument like a
    // HitExtension is a fairly tedious process, even if we'd like to do
    // something like printEvent(HitExtension ext), so just copy the extension,
    // if it exists, into some member variable and don't let the user hit
    // the print button until there's a hit with an extension present.
    
    *m_pExtension = hit.getExtension();
    
  } else {
    m_pPrintFit->setEnabled(false);
  }
}

//
// Private methods
//

//____________________________________________________________________________
/**
 * createHitBox
 *   Create and configure the hit group box containing widgets to display 
 *   basic hit information.
 *
 * @return QGroupBox* - pointer to the created QGroupBox object
 */
QGroupBox*
QHitData::createHitBox()
{
  QGroupBox* box = new QGroupBox;
  box->setTitle("Hit data");
  
  m_pId = new QLabel(QString("Crate: %1 Slot: %2 Channel: %3").arg(0).arg(0).arg(0));
  m_pRawData = new QLabel(QString("Energy: %1 Time: %2").arg(0).arg(0));
  
  QVBoxLayout* layout = new QVBoxLayout;
  layout->addWidget(m_pId);
  layout->addWidget(m_pRawData);
  box->setLayout(layout);

  return box;
}

//____________________________________________________________________________
/**
 * createHitBox
 *   Create and configure the classifier group box containing widgets to 
 *   display machine learning pulse classifier probabilities.
 *
 * @return QGroupBox* - pointer to the created QGroupBox object
 */
QGroupBox*
QHitData::createClassifierBox()
{
  QGroupBox* box = new QGroupBox;
  box->setTitle("Classifier data");

  m_pFit1Prob = new QLabel("Single pulse probability: N/A");
  m_pFit2Prob = new QLabel("Double pulse probability: N/A");

  QVBoxLayout* layout = new QVBoxLayout;
  layout->addWidget(m_pFit1Prob);
  layout->addWidget(m_pFit2Prob);
  box->setLayout(layout);

  return box;
}

//____________________________________________________________________________
/**
 * createFitBox
 *   Create and configure the fit group box containing widgets to select a 
 *   fit method and print fit results to stdout.
 *
 * @return QGroupBox* - pointer to the created QGroupBox object
 */
QGroupBox*
QHitData::createFitBox()
{
  QGroupBox* box = new QGroupBox;
  box->setTitle("Fit data");

  QLabel* label = new QLabel("Fit method:");
  m_pFitMethod = new QComboBox;
  m_pFitMethod->addItems({"Analytic", "Template"});
  m_pFitMethod->setCurrentIndex(1);
  m_pPrintFit = new QPushButton("Print");
  m_pPrintFit->setEnabled(false);
  
  QHBoxLayout* layout = new QHBoxLayout;
  layout->addWidget(label);
  layout->addWidget(m_pFitMethod);
  layout->addWidget(m_pPrintFit);
  box->setLayout(layout);

  return box;
}

//____________________________________________________________________________
/**
 * createConnections
 *   Create signal/slot connections for the hit data top widget. See Qt 
 *   documentation for more information.
 */
void
QHitData::createConnections()
{
  connect(m_pFitMethod, SIGNAL(currentIndexChanged(int)),
	  this, SLOT(configureFit()));
  connect(m_pPrintFit, SIGNAL(clicked()), this, SLOT(printFitResults()));
}

//____________________________________________________________________________
/**
 * updateHitData
 *   Update the data displayed in the hit data group box. Basic hit information
 *    contains at minimum and ID (crate/slot/channel), an energy and a time.
 *
 * @param hit - references the hit we are displaying data for
 */
void
QHitData::updateHitData(const DAQ::DDAS::DDASFitHit& hit)
{ 
  QString id = QString("Crate: %1 Slot: %2 Channel: %3").arg(hit.GetCrateID()).arg(hit.GetSlotID()).arg(hit.GetChannelID());
  QString data = QString("Energy: %1 Time: %2").arg(hit.GetEnergy()).arg(hit.GetTime());  
  m_pId->setText(id);
  m_pRawData->setText(data);
}

//____________________________________________________________________________
/**
 * configureFit
 *   Get the current text from the fit method box and pass it to the 
 *   FitManager to set which fitting functions to call when reconstructing 
 *   the fit data for display.
 */
void
QHitData::configureFit()
{  
  std::string method = m_pFitMethod->currentText().toStdString();
  m_pFitManager->configure(method);  
}

//____________________________________________________________________________
/**
 * printFitResults
 *   Print formatted fit results for the single and double pulse fits 
 *   to stdout.
 */
void
QHitData::printFitResults()
{
  std::cout << std::endl;
  std::cout << "#######################" << std::endl;
  std::cout << "##### Fit results #####" << std::endl;
  std::cout << "#######################" << std::endl;
  std::cout << "----- Single pulse -----" << std::endl;
  std::cout << std::left
	    << std::setw(6) << "Iter: "
	    << std::setw(2) << m_pExtension->onePulseFit.iterations
	    << std::setw(9) << " Status: "
	    << std::setw(2) << m_pExtension->onePulseFit.fitStatus 
	    << std::setw(8) << " Chisq: "
	    << std::setw(6) << m_pExtension->onePulseFit.chiSquare
	    << std::setw(9) << " Offset: "
	    << std::setw(8) << m_pExtension->onePulseFit.offset
	    << std::endl;
  
  std::cout << std::left
	    << std::setw(5) << "Amp: "
	    << std::setw(8) << m_pExtension->onePulseFit.pulse.amplitude
	    << std::setw(8) << " Steep: "
	    << std::setw(8) << m_pExtension->onePulseFit.pulse.steepness 
	    << std::setw(8) << " Decay: "
	    << std::setw(8) << m_pExtension->onePulseFit.pulse.decayTime 
	    << std::setw(6) << " Pos: "
	    << std::setw(8) << m_pExtension->onePulseFit.pulse.position
	    << std::endl;

  std::cout << std::endl;  
  std::cout << "----- Double pulse -----" << std::endl;
  std::cout << std::left
	    << std::setw(6) << "Iter: "
	    << std::setw(2) << m_pExtension->twoPulseFit.iterations
	    << std::setw(9) << " Status: "
	    << std::setw(2) << m_pExtension->twoPulseFit.fitStatus 
	    << std::setw(8) << " Chisq: "
	    << std::setw(6) << m_pExtension->twoPulseFit.chiSquare
	    << std::setw(9) << " Offset: "
	    << std::setw(8) << m_pExtension->twoPulseFit.offset
	    << std::endl;
  
  std::cout << std::left
	    << std::setw(5) << "Amp1: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[0].amplitude
	    << std::setw(8) << " Steep1: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[0].steepness 
	    << std::setw(8) << " Decay1: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[0].decayTime 
	    << std::setw(6) << " Pos1: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[0].position
	    << std::endl;
  
  std::cout << std::left
	    << std::setw(5) << "Amp2: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[1].amplitude
	    << std::setw(8) << " Steep2: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[1].steepness 
	    << std::setw(8) << " Decay2: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[1].decayTime 
	    << std::setw(6) << " Pos2: "
	    << std::setw(8) << m_pExtension->twoPulseFit.pulses[1].position
	    << std::endl;
}
