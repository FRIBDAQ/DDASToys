/** 
 * @file  QTraceView.cpp
 * @brief Implement Qt main application window class.
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
 * @brief Constructor.
 *
 * Create and configure the widgets which define the main window layout. 
 * Initialization list order and the order of the funciton calls in the 
 * constructor matter as some member variables depend on others being 
 * initialized to configure children or actions.
 *
 * @param parent  Pointer to QWidget parent object. Can be nullptr.
 */
QTraceView::QTraceView(QWidget* parent) :
  QWidget(parent), m_pDecoder(new DDASDecoder), m_pFitManager(new FitManager),
  m_config(false), m_templateConfig(false), m_fileName(""),
  m_pMenuBar(new QMenuBar), m_pFileMenu(nullptr), m_pOpenAction(nullptr),
  m_pExitAction(nullptr), m_pButtons{nullptr}, m_pSkipEvents(nullptr),
  m_pEventsToSkip(nullptr), m_pHitFilter{nullptr},
  m_pTopBoxes(createTopBoxes()), m_pHitData(new QHitData(m_pFitManager)),
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
  mainLayout->addWidget(m_pTopBoxes);
  mainLayout->addWidget(m_pHitData);
  mainLayout->addWidget(plotWidget);
  mainLayout->addWidget(m_pStatusBar);
  setLayout(mainLayout);

  createConnections();
  resetGUI();
  disableAll();
  
  m_pTimer->start(20);
}

//____________________________________________________________________________
/**
 * @brief Destructor.
 *
 * Qt _should_ manage deletion of child objects when each parent is destroyed
 * upon application exit. But we still should clean up our own stuff.
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
 * @brief Event handler for state changes. 
 *
 * Primary purpose here is to propagate the state changes to Root.
 *
 * @param e  Pointer to the handled QEvent.
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
 * @brief Create commands (user actions). 
 * 
 * See Qt documentation for more information. Actions for interactions with 
 * the top menu bar should be created here before adding them to their 
 * associated QMenu objects.
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
 * @brief Create and configure the top menu bar. 
 *
 * The menu bar is a collection of QMenu objects. Menu actions should be 
 * created prior to their addition.
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
 * @breif Create the top group box widgets.

 * These widgets contain the channel selection boxes and event handling.
 *
 * @return QWidget*  Pointer to the created QGroupBox object.
 */
QWidget*
QTraceView::createTopBoxes()
{
  QWidget* w = new QWidget;
  QHBoxLayout* mainLayout = new QHBoxLayout;

  // Channel selection
  
  QGroupBox* gb = new QGroupBox;
  gb->setTitle("Channel selection");
  QHBoxLayout* layout = new QHBoxLayout;
  const char* letext[3] = {"Crate:" , "Slot:", "Channel:"};
  for (int i=0; i<3; i++) {
    QLabel* l = new QLabel(letext[i]);
    m_pHitFilter[i] = new QLineEdit("*");
    layout->addWidget(l);
    layout->addWidget(m_pHitFilter[i]);
  }
  gb->setLayout(layout);

  // Skip control
  
  QGroupBox* gb2 = new QGroupBox;
  gb2->setTitle("Skip control");
  QHBoxLayout* layout2 = new QHBoxLayout;
  m_pSkipEvents = new QPushButton(tr("Skip"));
  m_pEventsToSkip = new QLineEdit("1");
  m_pEventsToSkip->setValidator(new QIntValidator(0, 999999, this));
  layout2->addWidget(m_pSkipEvents);
  layout2->addWidget(m_pEventsToSkip);
  gb2->setLayout(layout2);

  // Main control (next event, update, exit)
  
  QGroupBox* gb3 = new QGroupBox;
  gb3->setTitle("Main control");
  QHBoxLayout* layout3 = new QHBoxLayout;
  const char* btext[3] = {"Next", "Update", "Exit"};
  for (int i=0; i<3; i++) {
    m_pButtons[i] = new QPushButton(tr(btext[i]));
    layout3->addWidget(m_pButtons[i]);
  }
  gb3->setLayout(layout3);

  // Setup the actual widget
  
  mainLayout->addWidget(gb);
  mainLayout->addWidget(gb2);
  mainLayout->addWidget(gb3);
  w->setLayout(mainLayout);
  
  return w;
}

//____________________________________________________________________________
/**
 * @brief Create and configure the hit selection list widget.
 *
 * @return QListView*  Pointer to the created QListView widget.
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
 * @brief Create signal/slot connections for the main window. 
 *
 * See Qt signal/slot documentation for more information.
 */
void
QTraceView::createConnections()
{
  // Skip button
  connect(m_pSkipEvents, SIGNAL(clicked()), this, SLOT(skipEvents()));  
  
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
 * @brief Create the plotting widget.
 *
 * Combine the hit selection list and the Root canvas into a single widget 
 * with its own layout which can be added to the main window layout.
 *
 * @return QWidget*  Pointer to the created QWidget object.
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
 * @brief Set the status bar message.
 * 
 * @param msg  Message to display
 */
void
QTraceView::setStatusBar(std::string msg)
{
  m_pStatusBar->showMessage(tr(msg.c_str()));
}

//____________________________________________________________________________
/**
 * @brief Check if the current hit passes the hit filter. 
 *
 * Valid events match the crate/slot/channel values set in the filter boxes. 
 * Wildcard '*' characters pass everything. Valid hits must contain traces.
 *
 * @param hit  References the hit to validate.
 *
 * @return bool  True if the hit passes the filter, false otherwise.
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
 * @brief Update the hit selection list.
 * 
 * The list is updated based on the current list of filtered hits. Each hit is 
 * a formatted QString crate:slot:channel where the idenfiying information is 
 * read from the hit itself. The QStandardItemModel is used to provide data to 
 * the QListView interface, see Qt documentation for details.
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
 * @brief Reset and clear all GUI elements to default states.
 */
void
QTraceView::resetGUI()
{
  m_hits.clear();
  m_filteredHits.clear();
  updateSelectableHits();
  m_pRootCanvas->clear();
  m_pStatusBar->showMessage("");
}

//____________________________________________________________________________
/**
 * @brief Enable all UI buttons.
 */
void
QTraceView::enableAll()
{
  for (int i=0; i<3; i++) {
    m_pButtons[i]->setEnabled(true);
  }
  m_pSkipEvents->setEnabled(true);
  m_pEventsToSkip->setEnabled(true);
}

//____________________________________________________________________________
/** 
 * @brief Disable all UI buttons.
 *
 * Disable everything except exit, which we should always be able to do. Assume
 *  the exit button is the last one in the button list.
 */
void
QTraceView::disableAll()
{
  for (int i=0; i<2; i++) {
    m_pButtons[i]->setEnabled(false);
  }
  m_pSkipEvents->setEnabled(false);
  m_pEventsToSkip->setEnabled(false);
}

//____________________________________________________________________________
// Private slots

//____________________________________________________________________________
/**
 * @brief Open an NSCLDAQ event file.
 * 
 * Open a file using the QFileDialog and attempt to create a data source 
 * from it. Update status bar to show the currently loaded file and enable 
 * UI elements on the main window.
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
    m_pDecoder->createDataSource(m_fileName);
    resetGUI();
    setStatusBar(m_fileName);
    enableAll();
  }
}

//____________________________________________________________________________
/**
 * @brief Get the next event containing trace data.
 *
 * Get the next PHYSICS_EVENT event containing trace data from the data source
 * using the DDASDecoder. Apply the event filter and update the UI.
 */
void
QTraceView::getNextEvent()
{
  m_filteredHits.clear();
  
  while (m_filteredHits.empty()) {
    m_hits = m_pDecoder->getEvent();
    filterHits();
  }
  
  updateSelectableHits();
  m_pRootCanvas->clear();
  
  std::string msg = m_fileName + " -- Event " +
    std::to_string(m_pDecoder->getEventCount());
  setStatusBar(msg);
}

//____________________________________________________________________________
/** 
 * @brief Skip events in the source.
 *
 * The number of events to skip is read from the QLineEdit box when the skip 
 * button is clicked. If the end of the source file is encountered while 
 * skipping ahead, pop up a notification to that effect.
 */
void
QTraceView::skipEvents()
{
  std::string msg = "Skipping events, be patient...";
  setStatusBar(msg);
  int events = m_pEventsToSkip->text().toInt();
  int retval = m_pDecoder->skip(events);
  if (retval < 0) {
  } else {
    msg = "Successfully skipped " + std::to_string(events) + " PHYSICS_EVENTs. Hit 'Next' to display the next PHYSICS_EVENT containing trace data.";
    setStatusBar(msg);
  }
}

//____________________________________________________________________________
/**
 * @brief Apply the hit filter to the current list of hits for this event.
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
 * @brief Process a selected hit from the hit selection list. 
 *
 * The row number in the hit select list corresponds to the index of that hit 
 * in the list of filtered hits. Draw the trace and hit information and update
 * the hit data display.
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
 * @brief Process pending Root events.
 */
void
QTraceView::handleRootEvents()
{
   gSystem->ProcessEvents();
}

//____________________________________________________________________________
/**
 * @brief Test slot function which prints the current time to stdout. 
 */
void
QTraceView::test()
{
  std::time_t result = std::time(nullptr);
  std::cout << "Test slot call at: "
	    << std::asctime(std::localtime(&result))
	    << std::endl;
}
