#!/bin/tcsh

# Comment in the preferred compiler

#ifort -w -fast -parallel -o model inputmodel.f main.f rate12_complex_odes.f drive.f dvode.f subs.f shielding.f ismana.f
gfortran -O2 -w -std=legacy -o model inputmodel.f main.f rate12_complex_odes.f drive.f dvode.f subs.f shielding.f ismana.f



# 
# gfortran -c -O2 inputmodel.f -fbounds-check -fno-align-commons 
# gfortran -c -O2 main.f -fbounds-check -fno-align-commons 
# gfortran -c -O2 rate12_complex_odes.f -fbounds-check -fno-align-commons 
# gfortran -c -O2 drive.f -fbounds-check -fno-align-commons 	 
# gfortran -c -O2 dvode.f -fbounds-check -fno-align-commons 
# gfortran -c -O2 subs.f -fbounds-check -fno-align-commons 
# gfortran -c -O2 shielding.f -fbounds-check -fno-align-commons 
# gfortran -c -O2 ismana.f -fbounds-check -fno-align-commons 
# 
# gfortran -o model inputmodel.o main.o dvode.o rate12_complex_odes.o drive.o subs.o shielding.o ismana.o -fbounds-check -fno-align-commons  -Wimplicit-procedure -L/opt/local/lib
# 
# gfortran -o model inputmodel.o main.o dvode.o rate12_complex_odes.o drive.o subs.o shielding.o ismana.o -fbounds-check -fno-align-commons  -Wimplicit-procedure -L/opt/local/lib 
# 
# 
# gfortran -o model inputmodel.o main.o dvode.o rate12_complex_odes.o drive.o subs.o shielding.o ismana.o -fbounds-check -fno-align-commons -Wimplicit-procedure -L/opt/local/lib -mcmodel=large -march=armv8-a
# 
