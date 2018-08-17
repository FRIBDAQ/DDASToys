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
# @file plotUtils.tcl
# @brief Plotting utilities.
# @author Ron Fox <fox@nscl.msu.edu>
#

package require Plotchart

##
# addSeries
#   Given a plotchart plot, add a new series to it:
#
# @param plot   - the plot being added to.
# @param series - name of series
# @param color  - plot color
# @param xpts   - x coordinates of points.
# @param ypts   - y coordinate of points,.
#
proc addSeries {plot series color xpts ypts} {
    $plot dataconfig $series -color $color
    
    foreach x $xpts y $ypts {
        $plot plot $series $x $y
    }    
}

##
# plotDataSet
#   Plot a data set on a canvas using plotchart.
#  @param data - the data set.  The data are a dict with the following keys:
#                 * title - Plot /series title.
#                 * x     - Series x coordinates
#                 * y     - Series y coordinates.
#  @param canvas - The canvas on which to create/draw the plot.
proc plotDataSet {data canvas  {color black}} {
    set title [dict get $data title]
    
    # Figure out the plot limits and compute the axis specs:
    
    set xpts [dict get $data x]
    set ypts [dict get $data y]
    
    set xmin [expr min([join $xpts ,])]
    set xmax [expr max([join $xpts ,])]
    set xSpec [Plotchart::determineScale $xmin $xmax 0]
    
    set ymin [expr min([join $ypts ,])]
    set ymax [expr max([join $ypts ,])]
    set ymax [expr (1.05 + $ymax)];              # headroom.
    
    set ySpec [Plotchart::determineScale $ymin $ymax 0]

       
    # Create and plot the data:
    
    set plot [Plotchart::createXYPlot $canvas $xSpec $ySpec]
    $plot title $title
    $plot xtext Time
    
    addSeries $plot $title $color $xpts $ypts

    return $plot;       # in case user wants to annotate it.
    
}

##
# drawLine
#   Draw a line on a plot canvas given the world coordinates of the line
#   and the line's color.
#
# @param c  - Canvas on which to draw the line.
# @param x1,y1 - First x/y coordinates of the line in world coordinates.
# @param x2,y2 - Second x/y coordinates of the line in world coordinates.
# @param color - color of the line to add.
#
proc drawLine {c x1 y1 x2 y2 color} {
    set pt0 [::Plotchart::coordsToPixel $c $x1 $y1]
    set pt1 [::Plotchart::coordsToPixel $c $x2 $y2]
    
    $c create line [concat $pt0 $pt1] -fill $color
    
}

