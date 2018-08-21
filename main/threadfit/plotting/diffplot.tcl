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

if {[llength $argv] != 1} {
    puts stderr "Usage"
    puts stderr "   diffplot filename"
    exit -1
}

set dataSetNum 0
set fd [open $argv r]
set dataSet [readFitDataSet $fd]

set x [dict get $dataSet x]
set y1 [dict get $dataSet y1]
set y2 [dict get $dataSet y2]
set d  [dict get $dataSet diff]

# Figure out x range and y range.

set xmin 0
set xmax [lindex $x end]
set xSpec [Plotchart::determineScale $xmin $xmax]

set dmin [expr min([join $d ,])]
set dmax [expr max([join $d ,])]

set ymin [expr min([join $y1 ,])];
set ymax [expr max([join $y1 ,])]
set ymax2 [expr max([join $y2 ,])]
set ymax [expr max($ymax,$ymax2, $dmax)]
set ymax [expr 1.05*$ymax]
set ymin2 [expr min([join $y2 ,])]
set ymin [expr min($ymin, $ymin2, $dmin,  0)]


set yspec [Plotchart::determineScale $ymin $ymax 0]

if {[dict get $dataSet x] == [list]} break

canvas .c -width $::canvasWidth -height $::canvasHeight

set plot [Plotchart::createXYPlot .c $xSpec $yspec]
$plot title "Raw (black) fit (red) diff (blue)"

addSeries $plot Raw black $x $y2
addSeries $plot Fit red   $x $y1
addSeries $plot Diff blue $x [dict get $dataSet diff]



pack .c


    
