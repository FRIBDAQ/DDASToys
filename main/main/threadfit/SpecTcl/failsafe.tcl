#  SpecTclGUI save file created Wed May 30 16:00:05 EDT 2018
#  SpecTclGui Version: 1.0
#      Author: Ron Fox (fox@nscl.msu.edu)

#Tree params:

catch {treeparameter -create ChiSq1overChisq2 0 100 1000 {}}
treeparameter -setlimits ChiSq1overChisq2 0 100
treeparameter -setbins   ChiSq1overChisq2 1000
treeparameter -setunit   ChiSq1overChisq2 {}

catch {treeparameter -create Energy 0 16384 16384 {}}
treeparameter -setlimits Energy 0 16384
treeparameter -setbins   Energy 16384
treeparameter -setunit   Energy {}

catch {treeparameter -create event.raw.00 1 100 100 channels}
treeparameter -setlimits event.raw.00 1 100
treeparameter -setbins   event.raw.00 100
treeparameter -setunit   event.raw.00 channels

catch {treeparameter -create event.raw.01 1 100 100 channels}
treeparameter -setlimits event.raw.01 1 100
treeparameter -setbins   event.raw.01 100
treeparameter -setunit   event.raw.01 channels

catch {treeparameter -create event.raw.02 1 100 100 channels}
treeparameter -setlimits event.raw.02 1 100
treeparameter -setbins   event.raw.02 100
treeparameter -setunit   event.raw.02 channels

catch {treeparameter -create event.raw.03 1 100 100 channels}
treeparameter -setlimits event.raw.03 1 100
treeparameter -setbins   event.raw.03 100
treeparameter -setunit   event.raw.03 channels

catch {treeparameter -create event.raw.04 1 100 100 channels}
treeparameter -setlimits event.raw.04 1 100
treeparameter -setbins   event.raw.04 100
treeparameter -setunit   event.raw.04 channels

catch {treeparameter -create event.raw.05 1 100 100 channels}
treeparameter -setlimits event.raw.05 1 100
treeparameter -setbins   event.raw.05 100
treeparameter -setunit   event.raw.05 channels

catch {treeparameter -create event.raw.06 1 100 100 channels}
treeparameter -setlimits event.raw.06 1 100
treeparameter -setbins   event.raw.06 100
treeparameter -setunit   event.raw.06 channels

catch {treeparameter -create event.raw.07 1 100 100 channels}
treeparameter -setlimits event.raw.07 1 100
treeparameter -setbins   event.raw.07 100
treeparameter -setunit   event.raw.07 channels

catch {treeparameter -create event.raw.08 1 100 100 channels}
treeparameter -setlimits event.raw.08 1 100
treeparameter -setbins   event.raw.08 100
treeparameter -setunit   event.raw.08 channels

catch {treeparameter -create event.raw.09 1 100 100 channels}
treeparameter -setlimits event.raw.09 1 100
treeparameter -setbins   event.raw.09 100
treeparameter -setunit   event.raw.09 channels

catch {treeparameter -create event.sum 1 100 100 arbitrary}
treeparameter -setlimits event.sum 1 100
treeparameter -setbins   event.sum 100
treeparameter -setunit   event.sum arbitrary

catch {treeparameter -create fit1.Amplitude.0 0 8192 16384 {}}
treeparameter -setlimits fit1.Amplitude.0 0 8192
treeparameter -setbins   fit1.Amplitude.0 16384
treeparameter -setunit   fit1.Amplitude.0 {}

catch {treeparameter -create fit1.Amplitude.1 0 8192 16384 {}}
treeparameter -setlimits fit1.Amplitude.1 0 8192
treeparameter -setbins   fit1.Amplitude.1 16384
treeparameter -setunit   fit1.Amplitude.1 {}

catch {treeparameter -create fit1.Amplitude.2 0 8192 16384 {}}
treeparameter -setlimits fit1.Amplitude.2 0 8192
treeparameter -setbins   fit1.Amplitude.2 16384
treeparameter -setunit   fit1.Amplitude.2 {}

catch {treeparameter -create fit1.Amplitude.3 0 8192 16384 {}}
treeparameter -setlimits fit1.Amplitude.3 0 8192
treeparameter -setbins   fit1.Amplitude.3 16384
treeparameter -setunit   fit1.Amplitude.3 {}

catch {treeparameter -create fit1.Amplitude.4 0 8192 16384 {}}
treeparameter -setlimits fit1.Amplitude.4 0 8192
treeparameter -setbins   fit1.Amplitude.4 16384
treeparameter -setunit   fit1.Amplitude.4 {}

catch {treeparameter -create fit1.Chisquare.0 0 400 8000 {}}
treeparameter -setlimits fit1.Chisquare.0 0 400
treeparameter -setbins   fit1.Chisquare.0 8000
treeparameter -setunit   fit1.Chisquare.0 {}

catch {treeparameter -create fit1.Chisquare.1 0 400 8000 {}}
treeparameter -setlimits fit1.Chisquare.1 0 400
treeparameter -setbins   fit1.Chisquare.1 8000
treeparameter -setunit   fit1.Chisquare.1 {}

catch {treeparameter -create fit1.Chisquare.2 0 400 8000 {}}
treeparameter -setlimits fit1.Chisquare.2 0 400
treeparameter -setbins   fit1.Chisquare.2 8000
treeparameter -setunit   fit1.Chisquare.2 {}

catch {treeparameter -create fit1.Chisquare.3 0 400 8000 {}}
treeparameter -setlimits fit1.Chisquare.3 0 400
treeparameter -setbins   fit1.Chisquare.3 8000
treeparameter -setunit   fit1.Chisquare.3 {}

catch {treeparameter -create fit1.Chisquare.4 0 400 8000 {}}
treeparameter -setlimits fit1.Chisquare.4 0 400
treeparameter -setbins   fit1.Chisquare.4 8000
treeparameter -setunit   fit1.Chisquare.4 {}

catch {treeparameter -create fit1.iterations.0 0 50 50 Iterations}
treeparameter -setlimits fit1.iterations.0 0 50
treeparameter -setbins   fit1.iterations.0 50
treeparameter -setunit   fit1.iterations.0 Iterations

catch {treeparameter -create fit1.iterations.1 0 50 50 Iterations}
treeparameter -setlimits fit1.iterations.1 0 50
treeparameter -setbins   fit1.iterations.1 50
treeparameter -setunit   fit1.iterations.1 Iterations

catch {treeparameter -create fit1.iterations.2 0 50 50 Iterations}
treeparameter -setlimits fit1.iterations.2 0 50
treeparameter -setbins   fit1.iterations.2 50
treeparameter -setunit   fit1.iterations.2 Iterations

catch {treeparameter -create fit1.iterations.3 0 50 50 Iterations}
treeparameter -setlimits fit1.iterations.3 0 50
treeparameter -setbins   fit1.iterations.3 50
treeparameter -setunit   fit1.iterations.3 Iterations

catch {treeparameter -create fit1.iterations.4 0 50 50 Iterations}
treeparameter -setlimits fit1.iterations.4 0 50
treeparameter -setbins   fit1.iterations.4 50
treeparameter -setunit   fit1.iterations.4 Iterations

catch {treeparameter -create fit2.Amplitude1.0 0 8192 16834 {}}
treeparameter -setlimits fit2.Amplitude1.0 0 8192
treeparameter -setbins   fit2.Amplitude1.0 16834
treeparameter -setunit   fit2.Amplitude1.0 {}

catch {treeparameter -create fit2.Amplitude1.1 0 8192 16834 {}}
treeparameter -setlimits fit2.Amplitude1.1 0 8192
treeparameter -setbins   fit2.Amplitude1.1 16834
treeparameter -setunit   fit2.Amplitude1.1 {}

catch {treeparameter -create fit2.Amplitude1.2 0 8192 16834 {}}
treeparameter -setlimits fit2.Amplitude1.2 0 8192
treeparameter -setbins   fit2.Amplitude1.2 16834
treeparameter -setunit   fit2.Amplitude1.2 {}

catch {treeparameter -create fit2.Amplitude1.3 0 8192 16834 {}}
treeparameter -setlimits fit2.Amplitude1.3 0 8192
treeparameter -setbins   fit2.Amplitude1.3 16834
treeparameter -setunit   fit2.Amplitude1.3 {}

catch {treeparameter -create fit2.Amplitude1.4 0 8192 16834 {}}
treeparameter -setlimits fit2.Amplitude1.4 0 8192
treeparameter -setbins   fit2.Amplitude1.4 16834
treeparameter -setunit   fit2.Amplitude1.4 {}

catch {treeparameter -create fit2.Amplitude2.0 0 8192 16384 {}}
treeparameter -setlimits fit2.Amplitude2.0 0 8192
treeparameter -setbins   fit2.Amplitude2.0 16384
treeparameter -setunit   fit2.Amplitude2.0 {}

catch {treeparameter -create fit2.Amplitude2.1 0 8192 16384 {}}
treeparameter -setlimits fit2.Amplitude2.1 0 8192
treeparameter -setbins   fit2.Amplitude2.1 16384
treeparameter -setunit   fit2.Amplitude2.1 {}

catch {treeparameter -create fit2.Amplitude2.2 0 8192 16384 {}}
treeparameter -setlimits fit2.Amplitude2.2 0 8192
treeparameter -setbins   fit2.Amplitude2.2 16384
treeparameter -setunit   fit2.Amplitude2.2 {}

catch {treeparameter -create fit2.Amplitude2.3 0 8192 16384 {}}
treeparameter -setlimits fit2.Amplitude2.3 0 8192
treeparameter -setbins   fit2.Amplitude2.3 16384
treeparameter -setunit   fit2.Amplitude2.3 {}

catch {treeparameter -create fit2.Amplitude2.4 0 8192 16384 {}}
treeparameter -setlimits fit2.Amplitude2.4 0 8192
treeparameter -setbins   fit2.Amplitude2.4 16384
treeparameter -setunit   fit2.Amplitude2.4 {}

catch {treeparameter -create fit2.ChiSquare.0 0 400 8000 {}}
treeparameter -setlimits fit2.ChiSquare.0 0 400
treeparameter -setbins   fit2.ChiSquare.0 8000
treeparameter -setunit   fit2.ChiSquare.0 {}

catch {treeparameter -create fit2.ChiSquare.1 0 400 8000 {}}
treeparameter -setlimits fit2.ChiSquare.1 0 400
treeparameter -setbins   fit2.ChiSquare.1 8000
treeparameter -setunit   fit2.ChiSquare.1 {}

catch {treeparameter -create fit2.ChiSquare.2 0 400 8000 {}}
treeparameter -setlimits fit2.ChiSquare.2 0 400
treeparameter -setbins   fit2.ChiSquare.2 8000
treeparameter -setunit   fit2.ChiSquare.2 {}

catch {treeparameter -create fit2.ChiSquare.3 0 400 8000 {}}
treeparameter -setlimits fit2.ChiSquare.3 0 400
treeparameter -setbins   fit2.ChiSquare.3 8000
treeparameter -setunit   fit2.ChiSquare.3 {}

catch {treeparameter -create fit2.ChiSquare.4 0 400 8000 {}}
treeparameter -setlimits fit2.ChiSquare.4 0 400
treeparameter -setbins   fit2.ChiSquare.4 8000
treeparameter -setunit   fit2.ChiSquare.4 {}

catch {treeparameter -create fit2.dt.0 0 256 512 Sample}
treeparameter -setlimits fit2.dt.0 0 256
treeparameter -setbins   fit2.dt.0 512
treeparameter -setunit   fit2.dt.0 Sample

catch {treeparameter -create fit2.dt.1 0 256 512 Sample}
treeparameter -setlimits fit2.dt.1 0 256
treeparameter -setbins   fit2.dt.1 512
treeparameter -setunit   fit2.dt.1 Sample

catch {treeparameter -create fit2.dt.2 0 256 512 Sample}
treeparameter -setlimits fit2.dt.2 0 256
treeparameter -setbins   fit2.dt.2 512
treeparameter -setunit   fit2.dt.2 Sample

catch {treeparameter -create fit2.dt.3 0 256 512 Sample}
treeparameter -setlimits fit2.dt.3 0 256
treeparameter -setbins   fit2.dt.3 512
treeparameter -setunit   fit2.dt.3 Sample

catch {treeparameter -create fit2.dt.4 0 256 512 Sample}
treeparameter -setlimits fit2.dt.4 0 256
treeparameter -setbins   fit2.dt.4 512
treeparameter -setunit   fit2.dt.4 Sample

catch {treeparameter -create fit2.iterations.0 0 50 50 Iterations}
treeparameter -setlimits fit2.iterations.0 0 50
treeparameter -setbins   fit2.iterations.0 50
treeparameter -setunit   fit2.iterations.0 Iterations

catch {treeparameter -create fit2.iterations.1 0 50 50 Iterations}
treeparameter -setlimits fit2.iterations.1 0 50
treeparameter -setbins   fit2.iterations.1 50
treeparameter -setunit   fit2.iterations.1 Iterations

catch {treeparameter -create fit2.iterations.2 0 50 50 Iterations}
treeparameter -setlimits fit2.iterations.2 0 50
treeparameter -setbins   fit2.iterations.2 50
treeparameter -setunit   fit2.iterations.2 Iterations

catch {treeparameter -create fit2.iterations.3 0 50 50 Iterations}
treeparameter -setlimits fit2.iterations.3 0 50
treeparameter -setbins   fit2.iterations.3 50
treeparameter -setunit   fit2.iterations.3 Iterations

catch {treeparameter -create fit2.iterations.4 0 50 50 Iterations}
treeparameter -setlimits fit2.iterations.4 0 50
treeparameter -setbins   fit2.iterations.4 50
treeparameter -setunit   fit2.iterations.4 Iterations


# Pseudo parameter definitions


# Tree variable definitions:

treevariable -set vars.unused.00 0 furl/fort
treevariable -set vars.unused.01 0 furl/fort
treevariable -set vars.unused.02 0 furl/fort
treevariable -set vars.unused.03 0 furl/fort
treevariable -set vars.unused.04 0 furl/fort
treevariable -set vars.unused.05 0 furl/fort
treevariable -set vars.unused.06 0 furl/fort
treevariable -set vars.unused.07 0 furl/fort
treevariable -set vars.unused.08 0 furl/fort
treevariable -set vars.unused.09 0 furl/fort
treevariable -set vars.w1 1 arb/chan
treevariable -set vars.w2 1 arb/chan

# Gate definitions in reverse dependency order
 
gate DECAY s {fit2.dt.0 {5.500000 26.000000}}
gate FIT1.CHISQUARE s {fit1.Chisquare.0 {12.050000 383.850006}}
gate FIT2.CHISQUARE s {fit2.ChiSquare.0 {2.750000 11.650000}}
gate good * {FIT1.CHISQUARE FIT2.CHISQUARE}
gate goodratio s {ChiSq1overChisq2 {3.000000 30.200001}}

# Spectrum Definitions

catch {spectrum -delete DPP.E}
spectrum DPP.E 1 Energy {{0.000000 16384.000000 16384}} long
catch {spectrum -delete EvsDt}
spectrum EvsDt 2 {fit2.Amplitude2.0 fit2.dt.0} {{0.000000 8192.000000 8192} {0.000000 256.000000 512}} long
catch {spectrum -delete EvsDt-gated}
spectrum EvsDt-gated 2 {fit2.Amplitude2.0 fit2.dt.0} {{0.000000 8192.000000 8192} {0.000000 256.000000 512}} long
catch {spectrum -delete chisqratio}
spectrum chisqratio 1 ChiSq1overChisq2 {{0.000000 100.000000 1000}} long
catch {spectrum -delete fit1.Amplitude.0}
spectrum fit1.Amplitude.0 1 fit1.Amplitude.0 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit1.Amplitude.1}
spectrum fit1.Amplitude.1 1 fit1.Amplitude.1 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit1.Amplitude.2}
spectrum fit1.Amplitude.2 1 fit1.Amplitude.2 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit1.Amplitude.3}
spectrum fit1.Amplitude.3 1 fit1.Amplitude.3 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit1.Amplitude.4}
spectrum fit1.Amplitude.4 1 fit1.Amplitude.4 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit1.Chisquare.0}
spectrum fit1.Chisquare.0 1 fit1.Chisquare.0 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit1.Chisquare.1}
spectrum fit1.Chisquare.1 1 fit1.Chisquare.1 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit1.Chisquare.2}
spectrum fit1.Chisquare.2 1 fit1.Chisquare.2 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit1.Chisquare.3}
spectrum fit1.Chisquare.3 1 fit1.Chisquare.3 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit1.Chisquare.4}
spectrum fit1.Chisquare.4 1 fit1.Chisquare.4 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit1.iterations.0}
spectrum fit1.iterations.0 1 fit1.iterations.0 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit1.iterations.1}
spectrum fit1.iterations.1 1 fit1.iterations.1 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit1.iterations.2}
spectrum fit1.iterations.2 1 fit1.iterations.2 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit1.iterations.3}
spectrum fit1.iterations.3 1 fit1.iterations.3 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit1.iterations.4}
spectrum fit1.iterations.4 1 fit1.iterations.4 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit2.A0.0}
spectrum fit2.A0.0 1 fit2.Amplitude1.0 {{0.000000 8192.000000 16834}} long
catch {spectrum -delete fit2.A0.1}
spectrum fit2.A0.1 1 fit2.Amplitude1.1 {{0.000000 8192.000000 16834}} long
catch {spectrum -delete fit2.A0.2}
spectrum fit2.A0.2 1 fit2.Amplitude1.2 {{0.000000 8192.000000 16834}} long
catch {spectrum -delete fit2.A0.3}
spectrum fit2.A0.3 1 fit2.Amplitude1.3 {{0.000000 8192.000000 16834}} long
catch {spectrum -delete fit2.A0.4}
spectrum fit2.A0.4 1 fit2.Amplitude1.4 {{0.000000 8192.000000 16834}} long
catch {spectrum -delete fit2.A1.0}
spectrum fit2.A1.0 1 fit2.Amplitude2.0 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit2.A1.0!DECAY}
spectrum fit2.A1.0!DECAY 1 fit2.Amplitude2.0 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit2.A1.0!fit.CHISQUARE}
spectrum fit2.A1.0!fit.CHISQUARE 1 fit2.Amplitude2.0 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit2.A1.1}
spectrum fit2.A1.1 1 fit2.Amplitude2.1 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit2.A1.2}
spectrum fit2.A1.2 1 fit2.Amplitude2.2 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit2.A1.3}
spectrum fit2.A1.3 1 fit2.Amplitude2.3 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit2.A1.4}
spectrum fit2.A1.4 1 fit2.Amplitude2.4 {{0.000000 8192.000000 16384}} long
catch {spectrum -delete fit2.ChiSquare.0}
spectrum fit2.ChiSquare.0 1 fit2.ChiSquare.0 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit2.ChiSquare.1}
spectrum fit2.ChiSquare.1 1 fit2.ChiSquare.1 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit2.ChiSquare.2}
spectrum fit2.ChiSquare.2 1 fit2.ChiSquare.2 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit2.ChiSquare.3}
spectrum fit2.ChiSquare.3 1 fit2.ChiSquare.3 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit2.ChiSquare.4}
spectrum fit2.ChiSquare.4 1 fit2.ChiSquare.4 {{0.000000 400.000000 8000}} long
catch {spectrum -delete fit2.dt.0}
spectrum fit2.dt.0 1 fit2.dt.0 {{0.000000 256.000000 512}} long
catch {spectrum -delete fit2.dt.0!FITCHISQUARE}
spectrum fit2.dt.0!FITCHISQUARE 1 fit2.dt.0 {{0.000000 256.000000 512}} long
catch {spectrum -delete fit2.dt.1}
spectrum fit2.dt.1 1 fit2.dt.1 {{0.000000 256.000000 512}} long
catch {spectrum -delete fit2.dt.2}
spectrum fit2.dt.2 1 fit2.dt.2 {{0.000000 256.000000 512}} long
catch {spectrum -delete fit2.dt.3}
spectrum fit2.dt.3 1 fit2.dt.3 {{0.000000 256.000000 512}} long
catch {spectrum -delete fit2.dt.4}
spectrum fit2.dt.4 1 fit2.dt.4 {{0.000000 256.000000 512}} long
catch {spectrum -delete fit2.iterations.0}
spectrum fit2.iterations.0 1 fit2.iterations.0 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit2.iterations.1}
spectrum fit2.iterations.1 1 fit2.iterations.1 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit2.iterations.2}
spectrum fit2.iterations.2 1 fit2.iterations.2 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit2.iterations.3}
spectrum fit2.iterations.3 1 fit2.iterations.3 {{0.000000 50.000000 50}} long
catch {spectrum -delete fit2.iterations.4}
spectrum fit2.iterations.4 1 fit2.iterations.4 {{0.000000 50.000000 50}} long

# Gate Applications: 

apply good  EvsDt
apply goodratio  EvsDt-gated
apply DECAY  fit2.A1.0!DECAY
apply FIT1.CHISQUARE  fit2.A1.0!fit.CHISQUARE
apply FIT1.CHISQUARE  fit2.dt.0!FITCHISQUARE

#  filter definitions: ALL FILTERS ARE DISABLED!!!!!!!


#  - Parameter tab layout: 

set parameter(select) 1
set parameter(Array)  false

#-- Variable tab layout

set variable(select) 1
set variable(Array)  0
