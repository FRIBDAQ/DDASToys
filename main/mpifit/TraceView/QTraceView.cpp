#include "QTraceView.h"

#include <iostream>
#include <ctime>
#include <cstdint>

#include <QVBoxLayout>
#include <QStatusBar>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QComboBox>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QListView>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QItemSelectionModel>
#include <QTimer>
#include <QEvent>
#include <QWindowStateChangeEvent>

#include <TSystem.h>
#include <TCanvas.h>

#include <DDASFitHit.h>
#include <Configuration.h>
#include <fit_extensions.h>
#include "DDASDecoder.h"
#include "FitManager.h"
#include "QHitData.h"
#include "QRootCanvas.h"

// \TODO (ASC 2/2/23): Consider refactoring some of the widget elements if
// this class ends up pushing ~1000 lines

// \TODO (ASC 2/3/23): Management of canvas and hit selection list is not
// entirely clear - because they are added to the plotWidget, which is then
// added to this class, they should be re-parented and deleted automatically
// on close.

// Initialization list order and the order of the funciton calls in the
// constructor matter immensely as some member variables depend on others
// being initialized to configure children or actions.
QTraceView::QTraceView(QWidget* parent) :
  QWidget(parent),
  m_pDecoder(new DDASDecoder),
  m_pFitManager(new FitManager),
  m_count(0),
  m_config(false),
  m_templateConfig(false),
  m_fileName(""),
  m_pMenuBar(new QMenuBar),
  m_pFileMenu(nullptr),
  m_pOpenAction(nullptr),
  m_pExitAction(nullptr),
  m_pButtons{nullptr},
  m_pHitFilter{nullptr},
  m_pTopGroupBox(createTopGroupBox()),
  m_pHitData(new QHitData(m_pFitManager)),
  m_pHitSelectList(createHitSelectList()),  
  m_pRootCanvas(new QRootCanvas(m_pFitManager)),
  m_pTimer(new QTimer),
  m_pStatusBar(new QStatusBar)
{

  
  createActions();
  configureMenu();
  
  // Combine hit selection list and canvas into a single horizontally
  // aligned widget which we add to the main layout
  QWidget* plotWidget = createPlotWidget();  
  
  QVBoxLayout* mainLayout = new QVBoxLayout;
  mainLayout->setMenuBar(m_pMenuBar);
  mainLayout->addWidget(m_pTopGroupBox);
  mainLayout->addWidget(m_pHitData);
  mainLayout->addWidget(plotWidget);
  mainLayout->addWidget(m_pStatusBar);
  setLayout(mainLayout);

  createConnections();
  m_pStatusBar->showMessage(tr(""));  
  disableAll();
  
  m_pTimer->start(20);
}

QTraceView::~QTraceView()
{
  // Qt _should_ manage deletion of child objects when each parent is destroyed
  // upon application exit. But we still should clean up our own stuff.
  delete m_pDecoder;
  delete m_pFitManager;
  delete m_pHitData;
}

//
// Private member functions inherited from QWidget
// 

void
QTraceView::changeEvent(QEvent* e)
{
  if (e->type() == QEvent ::WindowStateChange) {
    QWindowStateChangeEvent* event = static_cast<QWindowStateChangeEvent*>(e);
    if ((event->oldState() & Qt::WindowMaximized) ||
	(event->oldState() & Qt::WindowMinimized) ||
	(event->oldState() == Qt::WindowNoState && 
	 this->windowState() == Qt::WindowMaximized)) {
      if (m_pRootCanvas->getCanvas()) {
	m_pRootCanvas->getCanvas()->Resize();
	m_pRootCanvas->getCanvas()->Update();
      }
    }
  }
}

//
// Private member functions
//

//
// Create and configure methods
//

// \TODO (ASC 2/3/23): Creation methods should not fail - there is an order in
// which (some of) these functions must be called but no safeguards to prevent
// someone from doing it wrong e.g. attempting to configure the menu bar before
// its created.

QGroupBox*
QTraceView::createTopGroupBox()
{
  QGroupBox* groupBox = new QGroupBox;
  QHBoxLayout* layout = new QHBoxLayout;

  const char* letext[3] = {"Crate:" , "Slot:", "Channel:"};
  for (int i=0; i<3; i++) {
    QLabel* l = new QLabel(letext[i]);
    m_pHitFilter[i] = new QLineEdit("*");
    layout->addWidget(l);
    layout->addWidget(m_pHitFilter[i]);
  }
  
  const char* btext[3] = {"Next", "Update", "Exit"};
  for (int i=0; i<3; i++) {
    m_pButtons[i] = new QPushButton(tr(btext[i]));
    layout->addWidget(m_pButtons[i]);
  }
  
  groupBox->setLayout(layout);
  
  return groupBox;
}

QListView*
QTraceView::createHitSelectList()
{    
  QListView* listView = new QListView;
  QStandardItemModel* model = new QStandardItemModel;
  listView->setEditTriggers(QAbstractItemView::NoEditTriggers);
  listView->setSelectionMode(QAbstractItemView::SingleSelection);
  listView->setModel(model);

  return listView;
}

void
QTraceView::createConnections()
{
  // Next button
  connect(m_pButtons[0], SIGNAL(clicked()), this, SLOT(getNextEvent()));
  // Update button
  connect(m_pButtons[1], SIGNAL(clicked()), this, SLOT(filterHits()));
  connect(m_pButtons[1], SIGNAL(clicked()), this, SLOT(updateSelectableHits()));
  // Exit button
  connect(m_pButtons[2], SIGNAL(clicked()), this, SLOT(close()));

  // Hit selection
  connect(m_pHitSelectList->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), this, SLOT(processHit()));

  // Timer to call inner loop of Root
  connect(m_pTimer, SIGNAL(timeout()), this, SLOT(handleRootEvents()));  
}

void
QTraceView::configureMenu()
{
  m_pFileMenu = new QMenu(tr("&File"), this);
  m_pFileMenu->addAction(m_pOpenAction);
  m_pFileMenu->addSeparator();
  m_pFileMenu->addAction(m_pExitAction);

  m_pMenuBar->addMenu(m_pFileMenu);
}

void
QTraceView::createActions()
{
  m_pOpenAction = new QAction(tr("&Open file..."), this);
  m_pOpenAction->setStatusTip(tr("Open an existing file"));
  connect(m_pOpenAction, SIGNAL(triggered()), this, SLOT(openFile()));

  m_pExitAction = new QAction(tr("E&xit"), this);
  m_pExitAction->setStatusTip(tr("Exit the application"));
  connect(m_pExitAction, &QAction::triggered, this, &QWidget::close);
}

QWidget*
QTraceView::createPlotWidget()
{
  // Set a label for the selection list
  QWidget* labeledList = new QWidget;
  QVBoxLayout* layout = new QVBoxLayout;
  QLabel* label = new QLabel("Crate:slot:channel hits with traces");
  layout->addWidget(label);
  layout->addWidget(m_pHitSelectList);
  labeledList->setLayout(layout);
  
  // Combine channel list and plot into a widget  
  QWidget* plot = new QWidget;
  QHBoxLayout* mainLayout = new QHBoxLayout;
  mainLayout->addWidget(labeledList);
  mainLayout->addWidget(m_pRootCanvas);
  plot->setLayout(mainLayout);

  return plot;
}

//
// Utilities
//

void
QTraceView::setStatusBar(std::string msg)
{
  m_pStatusBar->showMessage(tr(msg.c_str()));
}

bool
QTraceView::isValidHit(const DAQ::DDAS::DDASFitHit& hit)
{
  bool crateMatch = false;
  bool slotMatch = false;
  bool channelMatch = false;
  bool hasTrace = false;
  if ((m_pHitFilter[0]->text() == "*") ||
      (m_pHitFilter[0]->text().toUInt() == hit.GetCrateID())) {
    crateMatch = true;
  }
  if ((m_pHitFilter[1]->text() == "*") ||
      (m_pHitFilter[1]->text().toUInt() == hit.GetSlotID())) {
    slotMatch = true;
  }
  if ((m_pHitFilter[2]->text() == "*") ||
      (m_pHitFilter[2]->text().toUInt() == hit.GetChannelID())) {
    channelMatch = true;
  }

  std::vector<uint16_t> trace = hit.GetTrace();
  if (!trace.empty()) {
    hasTrace = true;
  }
  
  return (crateMatch && slotMatch && channelMatch && hasTrace);
}

void
QTraceView::updateSelectableHits()
{
  QStandardItemModel* model =
    reinterpret_cast<QStandardItemModel*>(m_pHitSelectList->model());
  model->clear();

  for (unsigned i=0; i<m_filteredHits.size(); i++) {
    // Qt 5.14+ supports arg(arg1, arg2, ...) but we're stuck with this
    QString id = QString("%1:%2:%3").arg(m_filteredHits[i].GetCrateID()).arg(m_filteredHits[i].GetSlotID()).arg(m_filteredHits[i].GetChannelID());
    QStandardItem* item = new QStandardItem(id);
    model->setItem(i, item);
  }
  
}

// A convienience for now but eventually we will want some control over
// how people interact with the UI.
void
QTraceView::enableAll()
{
  // Disable all buttons before exit, which we assume is last
  for (int i=0; i<2; i++) {
    m_pButtons[i]->setEnabled(true);
  }
}

void
QTraceView::disableAll()
{
  // Enable all buttons before exit, which we assume is last
  for (int i=0; i<2; i++) {
    m_pButtons[i]->setEnabled(false);
  }
}

//
// Private slots
//

void
QTraceView::openFile()
{
  QString fname = QFileDialog::getOpenFileName(this, tr("Open file"), "", tr("EVT files (*.evt);;All Files (*)"));
  
  m_fileName = fname.toStdString();
  m_fileName = "file://" + m_fileName;  
  
  if (fname.isEmpty()){
    std::cout << "WARNING: no file selected. Please open a file using File->Open file before continuing." << std::endl;
    return;
  } else {
    setStatusBar(m_fileName);
    m_pDecoder->createDataSource(m_fileName);
    enableAll();
  }
}

void
QTraceView::getNextEvent()
{
  m_filteredHits.clear();
  while (m_filteredHits.empty()) {
    m_hits = m_pDecoder->getEvent();
    filterHits();
    m_count++;
  }
  updateSelectableHits();
  m_pRootCanvas->clear();
  
  std::string msg = m_fileName + " -- Event " + std::to_string(m_count);
  setStatusBar(msg);
}

// List of hits with traces that match the crate/slot/channel ID specified
// in the filter boxes.
void
QTraceView::filterHits()
{
  if (!m_filteredHits.empty()) {
    m_filteredHits.clear();
  }
  
  // Check each hit and if the identifying information matches the filter
  // settings add it to the filtered hit
  for (auto& hit : m_hits) {
    if (isValidHit(hit)) {
      m_filteredHits.push_back(hit);
    }
  }
}

void
QTraceView::processHit()
{
  QItemSelectionModel* itemSelect = m_pHitSelectList->selectionModel();
  int idx = itemSelect->currentIndex().row();
  m_pRootCanvas->drawEvent(m_filteredHits[idx]);
  m_pHitData->update(m_filteredHits[idx]);  
}

// Call inner loop of Root
void
QTraceView::handleRootEvents()
{
   gSystem->ProcessEvents();
}

void
QTraceView::test()
{
  std::time_t result = std::time(nullptr);
  std::cout << "Test slot call at: "
	    << std::asctime(std::localtime(&result))
	    << std::endl;
}
