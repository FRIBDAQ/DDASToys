#include "QHitData.h"

#include <iostream>

#include <QVBoxLayout>
#include <QLabel>

#include <DDASFitHit.h>

QHitData::QHitData() 
{
  setTitle("Hit data");
  
  QVBoxLayout* mainLayout = new QVBoxLayout;
  m_rawHitData.s_pId = new QLabel(QString("Crate: %1 Slot: %2 Channel: %3").arg(0).arg(0).arg(0));
  m_rawHitData.s_pRawData = new QLabel(QString("Energy: %1 Time: %2").arg(0).arg(0));

  QVBoxLayout* rawLayout = new QVBoxLayout;
  QWidget* rawData = new QWidget;
  rawLayout->addWidget(m_rawHitData.s_pId);
  rawLayout->addWidget(m_rawHitData.s_pRawData);
  rawData->setLayout(rawLayout);

  mainLayout->addWidget(rawData);
  
  setLayout(mainLayout);
}

QHitData::~QHitData()
{}

void
QHitData::update(const DAQ::DDAS::DDASFitHit& hit)
{
  updateRawData(hit);
}

//
// Private methods
//

void
QHitData::updateRawData(const DAQ::DDAS::DDASFitHit& hit)
{
  // \TODO (ASC 2/2/23): Can you get the text and only reset part of it?
  // arg() returns a new string so we are making a whole bunch of calls to
  // the constructor every time we want to update.
  
  QString id = QString("Crate: %1 Slot: %2 Channel: %3").arg(hit.GetCrateID()).arg(hit.GetSlotID()).arg(hit.GetChannelID());
  QString data = QString("Energy: %1 Time: %2").arg(hit.GetEnergy()).arg(hit.GetTime());  
  m_rawHitData.s_pId->setText(id);
  m_rawHitData.s_pRawData->setText(data);
}
