/** @file: QTraceView.cpp
 *  @breif: Implement Qt main application window class
 */

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
#include "FitManager.h"
#include "DDASDecoder.h"
#include "QHitData.h"
#include "QRootCanvas.h"
#include <fit_extensions.h>

//____________________________________________________________________________
/**
 * Constructor
 *   Create and configure the widgets which define the main window layout. 
 *   Initialization list order and the order of the funciton calls in the 
 *   constructor matter as some member variables depend on others being 
 *   initialized to configure children or actions.
 *
 * @param parent - pointer to QWidget parent object, default = nullptr
 */
QTraceView::QTraceView(QWidget* parent) :
  QWidget(parent), m_pDecoder(new DDASDecoder), m_pFitManager(new FitManager),
  m_count(0), m_config(false), m_templateConfig(false), m_fileName(""),
  m_pMenuBar(new QMenuBar), m_pFileMenu(nullptr), m_pOpenAction(nullptr),
  m_pExitAction(nullptr), m_pButtons{nullptr}, m_pHitFilter{nullptr},
  m_pTopGroupBox(createTopGroupBox()), m_pHitData(new QHitData(m_pFitManager)),
  m_pHitSelectList(createHitSelectList()),
  m_pRootCanvas(new QRootCanvas(m_pFitManager)), m_pTimer(new QTimer),
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

//____________________________________________________________________________
/**
 * Destructor
 *   Qt _should_ manage deletion of child objects when each parent is destroyed
 *   upon application exit. But we still should clean up our own stuff.
 */
QTraceView::~QTraceView()
{
  delete m_pDecoder;
  delete m_pFitManager;
}

//
// Private member functions inherited from QWidget
// 

//____________________________________________________________________________
/**
 * changeEvent
 *   Event handler for state changes. Primary purpose here is to propagate 
 *   the state changes to Root.
 *
 * @param e - pointer to the handled QEvent
 */
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

//____________________________________________________________________________
// Create and configure methods

// \TODO (ASC 2/3/23): Creation methods should not fail without explanation -
// there is an order in which (some of) these functions must be called but no
// safeguards to prevent someone from doing it wrong e.g. attempting to
// configure the menu bar before its created.

//____________________________________________________________________________
/**
 * createActions
 *   Create commands (user actions). See Qt documentation for more information.
 *   Actions for interactions with the top menu bar should be created here 
 *   before adding them to their associated QMenu objects.
 */
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

//____________________________________________________________________________
/**
 * configureMenu
 *   Create and configure the top menu bar. The menu bar is a collection of 
 *   QMenu objects. Menu actions should be created prior to their addition.
 */
void
QTraceView::configureMenu()
{
  m_pFileMenu = new QMenu(tr("&File"), this);
  m_pFileMenu->addAction(m_pOpenAction);
  m_pFileMenu->addSeparator();
  m_pFileMenu->addAction(m_pExitAction);

  m_pMenuBar->addMenu(m_pFileMenu);
}

//____________________________________________________________________________
/**
 * createTopGroupBox
 *   Create the top group box widget containing the channel selection boxes 
 *   and event handling.
 *
 * @return QGroupBox* - pointer to the created QGroupBox object
 */
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

//____________________________________________________________________________
/**
 * createHitSelectList
 *  Create and configure the hit selection list widget.
 *
 * @return QListView* - a pointer to the created QListView widget
 */
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

//____________________________________________________________________________
/**
 * createConnections
 *   Create signal/slot connections for the main window. See Qt documentation 
 *   for more information.
 */
void
QTraceView::createConnections()
{
  // Next button
  
  connect(m_pButtons[0], SIGNAL(clicked()), this, SLOT(getNextEvent()));
  
  // Update buttons
  
  connect(m_pButtons[1], SIGNAL(clicked()), this, SLOT(filterHits()));
  connect(m_pButtons[1], SIGNAL(clicked()), this, SLOT(updateSelectableHits()));
  
  // Exit button
  
  connect(m_pButtons[2], SIGNAL(clicked()), this, SLOT(close()));

  // Hit selection
  
  connect(m_pHitSelectList->selectionModel(),
	  SIGNAL(selectionChanged(QItemSelection, QItemSelection)),
	  this, SLOT(processHit()));

  // Timer to call inner loop of Root
  
  connect(m_pTimer, SIGNAL(timeout()), this, SLOT(handleRootEvents()));  
}

//____________________________________________________________________________
/**
 * createPlotWidget
 *   Combine the hit selection list and the Root canvas into a single widget 
 *   with its own layout which can be added to the main window layout.
 *
 * @return QWidget* - pointer to the created QWidget object
 */
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

//____________________________________________________________________________
// Utilities

//____________________________________________________________________________
/**
 * setStatusBar
 *   Set the status bar message.
 * 
 * @param msg - message to display
 */
void
QTraceView::setStatusBar(std::string msg)
{
  m_pStatusBar->showMessage(tr(msg.c_str()));
}

//____________________________________________________________________________
/**
 * isValidHit
 *   Check if the current hit passes the hit filter. Valid events match the 
 *   crate/slot/channel values set in the filter boxes. Wildcard '*' characters
 *   pass everything. Valid hits must contain traces.
 *
 * @param hit - references the hit to validate
 *
 * @return bool - true if the hit passes the filter, false otherwise
 */
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

//____________________________________________________________________________
/**
 * updateSelectableHits
 *   Update the hit selection list based on the current list of filtered hits. 
 *   Each hit is a formatted QString crate:slot:channel where the idenfiying 
 *   information is read from the hit itself. The QStandardItemModel is used 
 *   to provide data to the QListView interface, see Qt documentation for 
 *   details.
 */
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

//____________________________________________________________________________
/**
 * enableAll
 *   Enable all UI buttons.
 */
void
QTraceView::enableAll()
{
  for (int i=0; i<3; i++) {
    m_pButtons[i]->setEnabled(true);
  }
}

//____________________________________________________________________________
/** 
 * disableAll
 *   Disable all UI buttons except for the exit button, which we assume is the 
 *   last one in the button list.
 */
void
QTraceView::disableAll()
{
  for (int i=0; i<2; i++) {
    m_pButtons[i]->setEnabled(false);
  }
}

//____________________________________________________________________________
// Private slots

//____________________________________________________________________________
/**
 * openFile
 *   Open a file using the QFileDialog and attempt to create a data source 
 *   from it. Update status bar to show the currently loaded file and enable 
 *   UI elements on the main window.
 */
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
    m_count = 0;
  }
}

//____________________________________________________________________________
/**
 * getNextEvent
 *   Get the next event from the data source using the DDASDecoder. Apply the 
 *   event filter and update the UI.
 */
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

//____________________________________________________________________________
/**
 * filterHits
 *   Apply the hit filter to the current list of hits for this event.
 */
void
QTraceView::filterHits()
{
  if (!m_filteredHits.empty()) {
    m_filteredHits.clear();
  }
  for (auto& hit : m_hits) {
    if (isValidHit(hit)) {
      m_filteredHits.push_back(hit);
    }
  }
}

//____________________________________________________________________________
/**
 * processHit
 *   Process a selected hit from the hit selection list. The row number in 
 *   the hit select list corresponds to the index of that hit in the list of 
 *   filtered hits. Draw the trace and hit information and update the hit data
 *   display.
 */
void
QTraceView::processHit()
{
  QItemSelectionModel* itemSelect = m_pHitSelectList->selectionModel();
  int idx = itemSelect->currentIndex().row();
  m_pRootCanvas->drawHit(m_filteredHits[idx]);
  m_pHitData->update(m_filteredHits[idx]);  
}

//____________________________________________________________________________
/**
 * handleRootEvents
 *   Process pending Root events.
 */
void
QTraceView::handleRootEvents()
{
   gSystem->ProcessEvents();
}

//____________________________________________________________________________
/**
 * test
 *   Test slot function which prints the current time to stdout. 
 */
void
QTraceView::test()
{
  std::time_t result = std::time(nullptr);
  std::cout << "Test slot call at: "
	    << std::asctime(std::localtime(&result))
	    << std::endl;
}
