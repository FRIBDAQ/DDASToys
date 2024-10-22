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
 * @file  QTraceView.cpp
 * @brief Implement Qt main application window class.
 */

#include "QTraceView.h"

#include <iostream>
#include <fstream>
#include <ctime>

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
#include <QCommandLineParser>
#include <QMessageBox>

#include <TSystem.h>
#include <TCanvas.h>

#include <DDASFitHit.h>
#include "FitManager.h"
#include "DDASDecoder.h"
#include "QHitData.h"
#include "QRootCanvas.h"

using namespace ddastoys;

//____________________________________________________________________________
/**
 * @details
 * Constructs the widgets which define the main window layout. Initialization 
 * list order and the order of the funciton calls in the constructor matter as 
 * some member variables depend on others being initialized to configure 
 * children or actions.
 */
QTraceView::QTraceView(QCommandLineParser& parser, QWidget* parent) :
    QWidget(parent), m_pDecoder(nullptr), m_pFitManager(nullptr),
    m_pMenuBar(nullptr), m_pFileMenu(nullptr), m_pOpenAction(nullptr),
    m_pExitAction(nullptr), m_pMainButtons{nullptr}, m_pSelectButtons{nullptr},
    m_pSelectLineEdit(nullptr), m_pHitFilter{nullptr}, m_pTopBoxes(nullptr),
    m_pHitData(nullptr), m_pHitSelectList(nullptr), m_pPlotWidget(nullptr),
    m_pRootCanvas(nullptr), m_pTimer(nullptr), m_pStatusBar(nullptr)
{
    m_pTimer      = new QTimer;
    m_pStatusBar  = new QStatusBar;
    
    m_pDecoder    = new DDASDecoder;    
    m_pFitManager = new FitManager;
    m_pHitData    = new QHitData(m_pFitManager);
    m_pRootCanvas = new QRootCanvas(m_pFitManager);
    
    createMenu();            // Creates m_pMenuBar widget (and actions).
    createTopBoxWidget();    // Creates m_pTopBoxes widget.
    createHitSelectWidget(); // Creates m_pHitSelectList widget.
    createPlotWidget();      // Creates m_pPlotWidget widget.
      
    // Combine hit selection list and canvas into a single horizontally
    // aligned widget which we add to the main layout:
      
    QVBoxLayout* mainLayout = new QVBoxLayout;
    mainLayout->setMenuBar(m_pMenuBar);
    mainLayout->addWidget(m_pTopBoxes);
    mainLayout->addWidget(m_pHitData);
    mainLayout->addWidget(m_pPlotWidget);
    mainLayout->addWidget(m_pStatusBar);
    setLayout(mainLayout);

    parseArgs(parser); // Assumes widget elements are configured.
  
    m_pTimer->start(20);
}

//____________________________________________________________________________
/**
 * @details
 * Qt _should_ manage deletion of child objects when each parent is destroyed
 * upon application exit. But we still should clean up our own stuff.
 */
QTraceView::~QTraceView()
{
    delete m_pDecoder;
    delete m_pFitManager;
}

//////////////////////////////////////////////////////////////////////////////
// Protected QWidget functions
/// 

//____________________________________________________________________________
/**
 * @details
 * Primary purpose here is to propagate the state changes to ROOT.
 */
void
QTraceView::changeEvent(QEvent* e)
{
    if (e->type() == QEvent ::WindowStateChange) {
	QWindowStateChangeEvent* event
	    = static_cast<QWindowStateChangeEvent*>(e);
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

//////////////////////////////////////////////////////////////////////////////
// Private slots
///

//____________________________________________________________________________
/**
 * @details
 * Open a file using the QFileDialog and attempt to create a data source 
 * from it. Update status bar to show the currently loaded file and enable 
 * UI elements on the main window.
 */
void
QTraceView::openFile()
{
    QString filename = QFileDialog::getOpenFileName(
	this, tr("Open File"), "", tr("Event files (*.evt);;All Files (*)")
	);

    // Get the current file path from the decoder. The path is an empty string
    // if no data source has been configured.  
    std::string currentPath = m_pDecoder->getFilePath();
  
    if (filename.isEmpty() && currentPath.empty()) {
	std::cout << "WARNING: no file selected."
		  << " Please open a file using File->Open file..."
		  << " before continuing." << std::endl;
	return;
    } else if (filename.isEmpty() && !currentPath.empty()) {
	std::cout << "WARNING: no file selected,"
		  << " but a data source already exists."
		  << " Current file path is: " << currentPath << std::endl;
    } else {
	configureSource(filename);
    }
}

//____________________________________________________________________________
/**
 * @details
 * The list is updated based on the current list of filtered hits. Each hit is 
 * a formatted QString crate:slot:channel where the idenfiying information is 
 * read from the hit itself. The QStandardItemModel is used to provide data to 
 * the QListView interface, see Qt documentation for details.
 */
void
QTraceView::updateSelectableHits()
{
    QStandardItemModel* model
	= reinterpret_cast<QStandardItemModel*>(m_pHitSelectList->model());
    model->clear();

    for (unsigned i = 0; i < m_filteredHits.size(); i++) {
	auto crate = m_filteredHits[i].getCrateID();
	auto slot = m_filteredHits[i].getSlotID();
	auto chan = m_filteredHits[i].getChannelID();
	// Qt 5.14+ supports arg(arg1, arg2, ...) but we're stuck with this:  
	QString id = QString("%1:%2:%3").arg(crate).arg(slot).arg(chan);    
	QStandardItem* item = new QStandardItem(id);
	model->setItem(i, item);
    }
}

//____________________________________________________________________________
/**
 * @details
 * Get the next PHYSICS_EVENT event from the data source using the DDASDecoder.
 * Apply the event filter and update the UI.
 * @note Depending on the channel selection filter settings and whether the hit
 * has trace data, the hit list may be empty.
 */
void
QTraceView::getNextEvent()
{
    m_hits = m_pDecoder->getEvent();

    // If the hit list is empty there are no more PHYSICS_EVENTs to grab
    // so issue the EOF warning and return from the function.    
    filterHits();
  
    if (m_hits.empty()) {
	issueEOFWarning();		
	return;
    }

    updateSelectableHits();
    m_pRootCanvas->clear();
  
    std::string msg = m_pDecoder->getFilePath()
	+ " -- Event " + std::to_string(m_pDecoder->getEventIndex());
    setStatusBar(msg);
}

//____________________________________________________________________________
/**
 * @details
 * Clear the list of filtered hits and keep grabbing events until the filtered 
 * hit list is not empty i.e., there is at least one valid hit according to 
 * the channel selection criteria.
 */
void
QTraceView::getNextEventWithTraces()
{    
    m_filteredHits.clear();  
    while (m_filteredHits.empty()) {
	getNextEvent();
    }
}

//____________________________________________________________________________
void
QTraceView::getNextEventFromList() {
    if (m_eventList.s_idx >= m_eventList.s_events.size()) {
	std::cerr << "No more events to view from list\n";
    } else {
	selectEventByIndex(m_eventList.s_events[m_eventList.s_idx]);
	m_eventList.s_idx++; // Next one.
    }
}

//____________________________________________________________________________
/** 
 * @details
 * The number of events to skip is read from the QLineEdit box when the skip 
 * button is clicked. If the end of the source file is encountered while 
 * skipping ahead, pop up a notification to that effect.
 */
void
QTraceView::skipEvents()
{
    int skipCount = m_pSelectLineEdit->text().toInt();
    setStatusBar("Skipping events, be patient...");  
    skipAndSelect(skipCount);
}

//____________________________________________________________________________
/** 
 * @details
 * The event index to select is read from the QLineEdit box when the select 
 * button is clicked. If the end of the source file is encountered while 
 * skipping ahead to the selected event, pop up a notification to that effect.
 */
void
QTraceView::selectEvent()
{
    int select = m_pSelectLineEdit->text().toInt();
    selectEventByIndex(select);
}

//____________________________________________________________________________
/**
 * @details
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
void
QTraceView::handleRootEvents()
{
    gSystem->ProcessEvents();
}

//____________________________________________________________________________
void
QTraceView::test()
{
    std::time_t result = std::time(nullptr);
    std::cout << "Test slot call at: "
	      << std::asctime(std::localtime(&result))
	      << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
// Private member functions
///

//____________________________________________________________________________
// Create and configure methods

/**
 * @todo (ASC 2/3/23): Creation methods should not fail without explanation -
 * there is an order in which (some of) these functions must be called but no
 * safeguards to prevent someone from doing it wrong e.g. attempting to
 * configure the menu bar before its created.
 */

//____________________________________________________________________________
/**
 * @detials
 * The menu bar is a collection of QMenu objects with associated signaling 
 * for actions. In this function we:
 * 1. Create the menu bar object.
 * 2. Create the file menu.
 * 3. Create actions for the file menu.
 * 4. Add actions to the file menu.
 * 5. Add the file menu to the menu bar.
 */
void
QTraceView::createMenu()
{
    m_pMenuBar = new QMenuBar;
    m_pFileMenu = new QMenu(tr("&File"), this);
    
    m_pOpenAction = new QAction(tr("&Open File..."), this);
    m_pOpenAction->setStatusTip(tr("Open an existing file"));
    connect(m_pOpenAction, SIGNAL(triggered()), this, SLOT(openFile()));

    m_pExitAction = new QAction(tr("E&xit"), this);
    m_pExitAction->setStatusTip(tr("Exit the application"));
    connect(m_pExitAction, &QAction::triggered, this, &QWidget::close);

    m_pFileMenu->addAction(m_pOpenAction);
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(m_pExitAction);
    m_pMenuBar->addMenu(m_pFileMenu);
}

//____________________________________________________________________________
/**
 * @details
 * These widgets contain the channel selection boxes and event handling.
 */
void
QTraceView::createTopBoxWidget()
{
    m_pTopBoxes = new QWidget;
    QHBoxLayout* mainLayout = new QHBoxLayout;

    // Channel selection
  
    QGroupBox* channelBox = new QGroupBox;
    channelBox->setTitle("Channel Selection");
    QHBoxLayout* channelBoxLayout = new QHBoxLayout;
    const char* letext[3] = {"Crate:" , "Slot:", "Channel:"};
    for (int i = 0; i < 3; i++) {
	QLabel* label = new QLabel(letext[i]);
	m_pHitFilter[i] = new QLineEdit("*");
	channelBoxLayout->addWidget(label);
	channelBoxLayout->addWidget(m_pHitFilter[i]);
    }
    channelBox->setLayout(channelBoxLayout);

    // Event selection control:
  
    QGroupBox* selectBox = new QGroupBox;
    selectBox->setTitle("Event Selection");
    QHBoxLayout* selectBoxLayout = new QHBoxLayout;
    const char* sbtext[2] = {"Skip", "Select"};
    for (int i = 0; i < 2; i++) {
	m_pSelectButtons[i] = new QPushButton(tr(sbtext[i]));
	selectBoxLayout->addWidget(m_pSelectButtons[i]);	
    }
    m_pSelectLineEdit = new QLineEdit("0");
    m_pSelectLineEdit->setValidator(new QIntValidator(0, 9999999, this));
    selectBoxLayout->addWidget(m_pSelectLineEdit);
    selectBox->setLayout(selectBoxLayout);

    // Main control (next event, update, exit):
  
    QGroupBox* mainBox = new QGroupBox;
    mainBox->setTitle("Main Control");
    QHBoxLayout* mainBoxLayout = new QHBoxLayout;
    const char* mbtext[3] = {"Next", "Update", "Exit"};
    for (int i = 0; i < 3; i++) {
	m_pMainButtons[i] = new QPushButton(tr(mbtext[i]));
	mainBoxLayout->addWidget(m_pMainButtons[i]);
    }
    mainBox->setLayout(mainBoxLayout);

    // Setup the actual widget:
  
    mainLayout->addWidget(channelBox);
    mainLayout->addWidget(selectBox);
    mainLayout->addWidget(mainBox);
    m_pTopBoxes->setLayout(mainLayout);
}

//____________________________________________________________________________
void
QTraceView::createHitSelectWidget()
{    
    m_pHitSelectList = new QListView;
    QStandardItemModel* model = new QStandardItemModel;
    m_pHitSelectList->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pHitSelectList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_pHitSelectList->setModel(model);
}

//____________________________________________________________________________
/**
 * @details 
 * Combine the hit selection list and the ROOT canvas into a single widget 
 * with its own layout which can be added to the main window layout.
 * We assume without checking that the ROOT canvas is not nullptr.
 */
void
QTraceView::createPlotWidget()
{
    // Requires that the hit selection is created first:
    
    if (!m_pHitSelectList) {
	createHitSelectWidget();
    }
    
    // Set a label for the selection list:
    
    QWidget* labeledList = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout;
    QLabel* label = new QLabel("Crate:Slot:Channel Hits with Traces");
    layout->addWidget(label);
    layout->addWidget(m_pHitSelectList);
    labeledList->setLayout(layout);
  
    // Combine channel list and plot into a widget:
    
    m_pPlotWidget = new QWidget;
    QHBoxLayout* mainLayout = new QHBoxLayout;
    mainLayout->addWidget(labeledList);
    mainLayout->addWidget(m_pRootCanvas);
    m_pPlotWidget->setLayout(mainLayout);
}

//____________________________________________________________________________
/**
 * @details
 * See Qt signal/slot documentation for more information. If an event list is 
 * specified at runtime, the function of some buttons changes slightly.
 */
void
QTraceView::createConnections(bool useEvtList)
{
    // Skip button
    connect(m_pSelectButtons[0], SIGNAL(clicked()), this, SLOT(skipEvents()));
    
    // Select button
    connect(m_pSelectButtons[1], SIGNAL(clicked()), this, SLOT(selectEvent()));
  
    // Next button.
    if (useEvtList) {
	connect(
	    m_pMainButtons[0], SIGNAL(clicked()),
	    this, SLOT(getNextEventFromList())
	    );
    } else {
	connect(
	    m_pMainButtons[0], SIGNAL(clicked()),
	    this, SLOT(getNextEventWithTraces())
	    );
    }

    // Update buttons  
    connect(m_pMainButtons[1], SIGNAL(clicked()), this, SLOT(filterHits()));
    connect(
	m_pMainButtons[1], SIGNAL(clicked()),
	this, SLOT(updateSelectableHits())
	);
  
    // Exit button  
    connect(m_pMainButtons[2], SIGNAL(clicked()), this, SLOT(close()));

    // Hit selection  
    connect(
	m_pHitSelectList->selectionModel(),
	SIGNAL(selectionChanged(QItemSelection, QItemSelection)),
	this, SLOT(processHit())
	);

    // Timer to call inner loop of ROOT  
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(handleRootEvents()));  
}

//____________________________________________________________________________
/**
 * @details
 * Create a data source from a filename string. Update status bar to show the
 * currently loaded file and enable UI elements on the main window. Errors 
 * which occur during data source creation are dealt with in the decoder.
 */
void
QTraceView::configureSource(QString filename)
{
    m_pDecoder->createDataSource(filename.toStdString());
    reset();
    setStatusBar(filename.toStdString());
    enableGUI(true);
}

//____________________________________________________________________________
void
QTraceView::skipAndSelect(int count)
{
    int retval = m_pDecoder->skip(count);    
    if (retval < 0) {   
	issueEOFWarning();
    } else {
	getNextEvent();
    }
}

//____________________________________________________________________________
/**
 * @details
 * Events are zero-indexed.
 */ 
void
QTraceView::selectEventByIndex(int idx)
{
    int current = m_pDecoder->getEventIndex();

    // We're currently looking at this event:

    if (idx == current) {
	return;
    }
    
    int skipCount = idx - current - 1; // Our idx is next event.
    
    // We've advanced in the file and are trying to go backwards.
    // Issue a warning and don't do anything:
    
    if (current >= 0 && skipCount < 0) {
	issueWarning(
	    "Current event = " + std::to_string(current)
	    + ", selected event = " + std::to_string(idx)
	    + "\nCannot select previous events!\n"
	    );
	return;
    }

    skipAndSelect(skipCount);    
}

//____________________________________________________________________________
/**
 * @details
 * Valid events match the crate/slot/channel values set in the filter boxes. 
 * Wildcard '*' characters pass everything. Valid hits must contain traces.
 */
bool
QTraceView::isValidHit(const DDASFitHit& hit)
{
    bool crateMatch = false;
    bool slotMatch = false;
    bool channelMatch = false;
    bool hasTrace = false;
    if ((m_pHitFilter[0]->text() == "*") ||
	(m_pHitFilter[0]->text().toUInt() == hit.getCrateID())) {
	crateMatch = true;
    }
    if ((m_pHitFilter[1]->text() == "*") ||
	(m_pHitFilter[1]->text().toUInt() == hit.getSlotID())) {
	slotMatch = true;
    }
    if ((m_pHitFilter[2]->text() == "*") ||
	(m_pHitFilter[2]->text().toUInt() == hit.getChannelID())) {
	channelMatch = true;
    }

    std::vector<uint16_t> trace = hit.getTrace();
    if (!trace.empty()) {
	hasTrace = true;
    }
  
    return (crateMatch && slotMatch && channelMatch && hasTrace);
}

//____________________________________________________________________________
// Utilities

//____________________________________________________________________________
void
QTraceView::setStatusBar(std::string msg)
{
    m_pStatusBar->showMessage(tr(msg.c_str()));
}

//____________________________________________________________________________
void
QTraceView::reset()
{
    m_hits.clear();
    m_filteredHits.clear();
    updateSelectableHits();
    m_pRootCanvas->clear();
    m_pStatusBar->showMessage("");
}

//____________________________________________________________________________
void
QTraceView::enableGUI(bool status)
{
    enableMainGUI(status);
    enableSelectGUI(status);
}

//____________________________________________________________________________
void
QTraceView::enableMainGUI(bool status)
{
    for (int i = 0; i < 2; i++) {
	m_pMainButtons[i]->setEnabled(status);
    }
    m_pMainButtons[2]->setEnabled(true); // Exit always enabled.
}

//____________________________________________________________________________
void
QTraceView::enableSelectGUI(bool status)
{
    for (int i = 0; i < 2; i++) {
	m_pSelectButtons[i]->setEnabled(status);
    }
    m_pSelectLineEdit->setEnabled(status);
}

//____________________________________________________________________________
/** 
 * @details
 * The warning is issued as a modal dialog, blocking until the user closes it.
 */
void
QTraceView::issueWarning(std::string msg)
{
    QMessageBox msgBox;
    msgBox.setText(QString::fromStdString(msg));
    msgBox.setIcon(QMessageBox::Warning);
    msgBox.exec();
}

/** 
 * @details
 * Create the EOF message and issue the warning.
 */
void
QTraceView::issueEOFWarning()
{
    m_hits.clear();
    m_filteredHits.clear();
    updateSelectableHits();
    m_pRootCanvas->clear();
    
    std::string msg("No more physics events in this file.");
    msg += "The file contains ";
    msg += std::to_string(m_pDecoder->getEventCount());
    msg += " physics events.";
    
    issueWarning(msg);
}

//____________________________________________________________________________
/**
 * @details
 * Use QCommandLineParser to read in and parse positional arguments supplied 
 * on the command line when the program is run. All arguments are considered 
 * optional. Handle the arguments which are provided and issue error messages 
 * if necessary.
 *
 * Some optional arguments may slightly change the function of the GUI 
 * elements. If an event list is specified by the user at runtime, the 'Next' 
 * button is used to select the next entry from the event list, not the next 
 * channel with traces. For simplicity, at the time being skipping and 
 * selecting specific events from the event list is not supported, so the 
 * 'Skip' and 'Select' buttons on the hit selection box are disabled.
 *
 * @note An event list can only be specified at runtime, and cannot be reloaded
 * or modified once the program is running. Restarting the program is required 
 * if the event list is modified or you wish to run `traceview` over all events.
 *
 * @todo (ASC 10/22/24): The event list should also be configurable from the 
 * main menu or some other GUI element as with other optional parameters. 
 */
void
QTraceView::parseArgs(QCommandLineParser& parser)
{
    const QStringList args = parser.positionalArguments();
    const QStringList names = parser.optionNames();

    bool sourceSet = parser.isSet("source");
    if (sourceSet) {
	QString filename = parser.value("source");
	configureSource(filename); // Also enables GUI elements.
    } else {
	enableGUI(false);
    }

    bool methodSet = parser.isSet("method");
    if (methodSet) {
	QString method = parser.value("method");
	try {
	    m_pHitData->setFitMethod(method);
	} catch (std::invalid_argument& e) {
	    std::cerr << "QTraceView::parseArgs(): Unknown fitting method "
		      << method.toStdString() << " read from command line."
		      << " Setting fit method to 'Analytic'." << std::endl;
	    m_pHitData->setFitMethod("Analytic");
	}
    }
    bool eventListSet = parser.isSet("event-list");
    if (eventListSet) {
	QString eventListFile = parser.value("event-list");
	parseEventList(eventListFile);
	enableSelectGUI(false); // Disable 'Skip', 'Select', QLineEdit
    }

    // Signals/slots depend on whether we have an event list or not:
    
    createConnections(eventListSet);
}

/**
 * @details
 * Events can be specified in the event list file in any order and duplicate 
 * entries are removed. Event indices are expected to be >= 0, negative
 * indices will be ignored.
 */
void
QTraceView::parseEventList(QString fname)
{
    std::ifstream fin;
    fin.open(fname.toStdString());
    
    // Assume one event per line:
    
    int evt;
    std::vector<int> tmp;
    
    while (fin >> evt) {
	if (evt < 0) {
	    std::cerr << "Negative event index " << evt
		      << " found when parsing event file "
		      << fname.toStdString() << "! Ignoring..."
		      << std::endl;		
	} else {
	    tmp.push_back(evt);
	}
    }

    // Uniquify, sort, and assign:
    
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());    

    m_eventList = EventList(tmp);
}
