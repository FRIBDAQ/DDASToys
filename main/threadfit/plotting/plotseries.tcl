#!/bin/sh
# -*- tcl -*-
# The next line is executed by /bin/sh, but not tcl \
exec tclsh "$0" ${1+"$@"}

#    This software is Copyright by the Board of Trustees of Michigan
#    State University (c) Copyright 2014.
#
#    You may use this software under the terms of the GNU public license
#    (GPL).  The terms of this license are described at:
#
#     http://www.gnu.org/licenses/gpl.txt
#
#    Authors:
#             Ron Fox
#             Jeromy Tompkins 
#	     NSCL
#	     Michigan State University
#	     East Lansing, MI 48824-1321



##
# @file rawplot.tcl
# @brief Plot data in simple text files
# @author Ron Fox <fox@nscl.msu.edu>
#

package require Tk
source utilities.tcl
source plotUtils.tcl

##
# Plots data in text files that are of the form
#   Title
#
#    x   y
# ...
#  Either EOF or another for another data set.
#  Each data set, for now, is plotted in its own canvas.
#  The canvases are vertically stacked in a single toplevel

#  Default canvas size
set canvasWidth    800
set canvasHeight   500

# override fromt env vars if defined

if {[array names env width] ne ""} {
    set canvasWidth $env(width)
}
if {[array names env height] ne ""} {
    set canvasHeight $env(height)
}

#  the name of the file to plot is on the command line:

if {[llength $argv] != 2} {
    puts stderr "Usage"
    puts stderr "   plotseries filename index"
    exit -1
}

set dataSetNum 0
set fname [lindex $argv 0]
set index [lindex $argv 1]

set fd [open $fname r]
while {![eof $fd]} {
    set dataSet [readDataSet $fd $index]
    
    if {[dict get $dataSet x] == [list]} break

    set canvasName .c$dataSetNum
    canvas .c$dataSetNum -width $::canvasWidth -height $::canvasHeight
    plotDataSet $dataSet .c$dataSetNum
    pack .c$dataSetNum
    
    incr dataSetNum
}
    
