/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  QHitData.cpp
 * @brief Implementation of hit data management class.
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

using namespace ddastoys;

//____________________________________________________________________________
/**
 * @details
 * Constructs QHitData widgets and defines their layout. This class does not
 * manage the FitManager object.
 */
QHitData::QHitData(FitManager* pFitMgr, QWidget* parent) :
    QWidget(parent), m_pFitManager(pFitMgr), m_pExtension(nullptr)
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

    m_pExtension = new HitExtension;
}

//____________________________________________________________________________
/**
 * @details
 * Destruction of FitManager is left to the caller which owns it.
 */
QHitData::~QHitData()
{}

//____________________________________________________________________________
void
QHitData::update(const DDASFitHit& hit)
{
    updateHitData(hit);
    if (hit.hasExtension()) {
	m_pPrintFit->setEnabled(true);

	// Calling a slot function with an non-standard type argument like a
	// HitExtension is a fairly tedious process, even if we'd like to do
	// something like printEvent(HitExtension ext), so just copy the
	// extension, if it exists, into some member variable and don't let
	// the user hit the print button until there's a hit with an extension
	// present.
    
	*m_pExtension = hit.getExtension();
    
    } else {
	m_pPrintFit->setEnabled(false);
    }
}

//____________________________________________________________________________
/**
 * @details
 * Set the fit method from a QString if a method parameter is supplied on
 * the command line. Throw an exception if the fit method is not recognized.
 */
void
QHitData::setFitMethod(QString method)
{
    // Get the combo box index from the input string. Case-insensitive unless
    // Qt::MatchCaseSensitive flag is also specified. Returns the index of the
    // item containing the method text otherwise returns -1.
  
    int idx = m_pFitMethod->findText(method, Qt::MatchFixedString);

    if (idx < 0) {
	std::string msg = "Input '" + method.toStdString() + "' does not "
	    "match any known fit method. Please ensure that the fitting "
	    "method is properly configured before continuing.";
	throw std::invalid_argument(msg);
    } else {
	m_pFitMethod->setCurrentIndex(idx);
    }
  
}

//
// Private methods
//

//____________________________________________________________________________
QGroupBox*
QHitData::createHitBox()
{
    QGroupBox* box = new QGroupBox;
    box->setTitle("Hit Data");
  
    m_pId = new QLabel(
	QString("Crate: %1 Slot: %2 Channel: %3"
	    ).arg(0).arg(0).arg(0));
    m_pRawData = new QLabel(QString("Energy: %1 Time: %2").arg(0).arg(0));
  
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(m_pId);
    layout->addWidget(m_pRawData);
    box->setLayout(layout);

    return box;
}

//____________________________________________________________________________
QGroupBox*
QHitData::createClassifierBox()
{
    QGroupBox* box = new QGroupBox;
    box->setTitle("Classifier Data");

    m_pFit1Prob = new QLabel("Single pulse probability: N/A");
    m_pFit2Prob = new QLabel("Double pulse probability: N/A");

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(m_pFit1Prob);
    layout->addWidget(m_pFit2Prob);
    box->setLayout(layout);

    return box;
}

//____________________________________________________________________________
QGroupBox*
QHitData::createFitBox()
{
    QGroupBox* box = new QGroupBox;
    box->setTitle("Fit Data");

    QLabel* label = new QLabel("Method:");
    m_pFitMethod = new QComboBox;
    m_pFitMethod->addItems({"Analytic", "Template", "ML_Inference"});
    m_pFitMethod->setCurrentIndex(0);
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
void
QHitData::createConnections()
{
    connect(
	m_pFitMethod, SIGNAL(currentIndexChanged(int)),
	this, SLOT(configureFit())
	);
    connect(m_pPrintFit, SIGNAL(clicked()), this, SLOT(printFitResults()));
}

//____________________________________________________________________________
/**
 * @details
 * This expects a hit from the new-style fit extension which contains the 
 * single and double pulse probabilities calculated by the machine learning 
 * inference code.
 */
void
QHitData::updateHitData(const DDASFitHit& hit)
{
    // All hits have an ID and hit data:
    
    QString id = QString(
	"Crate: %1 Slot: %2 Channel: %3"
	).arg(hit.getCrateID()).arg(hit.getSlotID()).arg(hit.getChannelID());

    // Display 3 decimal places of the double-precision full timestamp
    // possibly with the CFD correction. No limits on the width:
    
    QString hitData = QString(
	"Energy: %1 Time: %2"
	).arg(hit.getEnergy()).arg(hit.getTime(), 0, 'f', 3);

    // If there's a fit, update the probabilities:
    
    QString fit1Prob = QString("Single pulse probability: N/A");
    QString fit2Prob = QString("Double pulse probability: N/A");
    if (hit.hasExtension()) {
	auto ext = hit.getExtension();
	fit1Prob = QString("Single pulse probability: %1").arg(ext.singleProb);
	fit2Prob = QString("Double pulse probability: %1").arg(ext.doubleProb);
    }

    m_pId->setText(id);
    m_pRawData->setText(hitData);
    m_pFit1Prob->setText(fit1Prob);
    m_pFit2Prob->setText(fit2Prob);
}

//____________________________________________________________________________
/**
 * @details
 * Get the current text from the fit method combo box and pass it to the 
 * FitManager. The FitManager sets which fitting functions to call when 
 * reconstructing the fit data for display.
 */
void
QHitData::configureFit()
{  
    std::string method = m_pFitMethod->currentText().toStdString();
    m_pFitManager->configure(method);  
}

//____________________________________________________________________________
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
    
    std::cout << std::endl;  
    std::cout << "----- Classification -----" << std::endl;
    std::cout << std::left
	      << std::setw(5) << "P(single): "
	      << std::setw(8) << m_pExtension->singleProb
	      << "\n"
	      << std::setw(5) << "P(double): "
	      << std::setw(8) << m_pExtension->doubleProb
	      << std::endl;
}
