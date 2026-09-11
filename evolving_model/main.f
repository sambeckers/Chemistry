      SUBROUTINE MAIN(DENS,TEMPGAS,TEMPDUST,
     1   AV,ZETACR,ZETAXR,TIME,IPOINT,TTOT,ANA,IANA,ANAFILE,
     2   REACFILE,SPECFILE,BINDFILE,GRAINFILE,RADFILE,SWITCHFILE,
     3   GISM,MASS,Y,ABUN,SPEC,NSPEC,NTOT)   

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Declaration of variables
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      IMPLICIT NONE
      
      INTEGER I,J

C I,J = counters  

      INTEGER D,E,G,H            
      PARAMETER (D=1600)
      PARAMETER (G=1000)
      PARAMETER (H=10000)

C D = max number of grid points
C F = max number of time steps
C G = max number of species
C H = max number of reactions
 
      DOUBLE PRECISION XCOORD,YCOORD,ZCOORD
      DOUBLE PRECISION DENS,TEMPGAS,TEMPDUST
      DOUBLE PRECISION GFUV,ZETACR,ZETAXR

C XCOORD, YCOORD, ZCOORD = model coordinates
C DENS = H nuclei density (cm-3)
C TEMPGAS =  gas temperature (K)
C TEMPDUST = dust temperature (K)
C GFUV = wavelength-integrated UV field (erg cm-2 s-1) 
C ZETACR = cosmic-ray ionisation rate (s-1)
C ZETAXR = x-ray ionisation rates (s-1)

      CHARACTER*60 REACFILE,SPECFILE,BINDFILE
      CHARACTER*60 GRAINFILE,RADFILE,SWITCHFILE
      CHARACTER*60 OUTFILE,ANAFILE

C REACFILE = reaction rates filename
C SPECFILE = species filename
C BINDFILE = binding energy filename
C GRAINFILE = grain parameters filename
C SWITCHFILE = reaction parameters filename
C RADFILE = radiation field parameters filename
C OUTFILE = output filename
C ANAFILE = analyse output filename
                    
      INTEGER NREAC,IPOINT,IANA,NSWITCH
      INTEGER LOWTEMP(H),UPTEMP(H),RTYPE(H),SWITCH(20)
      DOUBLE PRECISION ALPHA(H),BETA(H),GAMMA(H)
      DOUBLE PRECISION K(H)
      CHARACTER*10 RE1(H),RE2(H),RE3(H)
      CHARACTER*10 P1(H),P2(H),P3(H),P4(H),P5(H)
      
      LOGICAL ANA

C NREAC = number of reactions
C IPOINT = current grid point
C IANA - grid point at which to run ANALYSE subroutine
C LOW/UPTEMP = lower/upper temperature bound for rate coefficient (K)
C RTYPE = reaction type (see README file)
C ALPHA,BETA,GAMMA = parameters for reaction rate coefficient (see README file)
C K = reaction rate coefficients (see README file)
C REx = reactants
C Px = products
C NSWITCH = number of reactions to switch off
C SWITCH = array containing reaction types to switch off
    
      INTEGER NSPEC,NICE
      DOUBLE PRECISION Y(G)
      DOUBLE PRECISION TIME,TTOT,ABUN(G)
      DOUBLE PRECISION MASS(G),BIND(G),YIELD(G)
      CHARACTER*10 SPEC(G),SICE(G)

C NSPEC = total number of species
C NICE = number of ice species
C Y = species abundances (cm-3)
C ABUN = radial/time-dependent abundances (relative to Hnuc density)
C MASS = species mass (amu)
C BIND = species binding energy (ice only; K)
C YIELD = photodesorption yield (ice only; molecules photon-1)
C SPEC = species names
C SICE = ice species names (for binding energies and grain reactions)

      INTEGER NCON,NELEM,NTOT
      PARAMETER(NELEM=10)
      DOUBLE PRECISION X(1),TOTAL,FRAC(NELEM)
      CHARACTER*2 ELEM(NELEM)
      
C NCON = number of conserved species (electrons only)
C NELEM = number of elements
C NTOT = total number of species
C X = initalise electron abundance
C TOTAL = total charge (conserved, hence, 0)  
C FRAC = fractional abundance of elements
C ELEM = element names (important for X-ray ionisation rates)    
      
      DOUBLE PRECISION TSTART,TFINAL,T0

C TSTART = first time step for integration (yr or sec)
C TFINAL = final time step (yr or sec)
C T0 = time counter (reset to 0)

      INTEGER NFUV
      DOUBLE PRECISION GISM,HABING
      DOUBLE PRECISION ZISM,CRPHOT,UVPHOT
      DOUBLE PRECISION AV,SHIELD
      DOUBLE PRECISION N_H2,N_CO,N_N2

C GISM = interstellar radiation field (erg cm-2 s-1)
C HABING = habing field (photons cm-2 s-1)
C ZISM = cosmic-ray ionisation rate (s-1)
C CRPHOT = cosmic-ray-induced photon flux (photons cm-2 s-1)
C UVPHOT = total UV photon flux (photons cm-2 s-1)
C AV = visual extinction (mag)
C SHIELD = shielding function
C N_x = column densities for self and mutual shielding (cm2)

      DOUBLE PRECISION HLOSS,STICK
      DOUBLE PRECISION A1,A2,A3,B1,B2,ECHEM
      DOUBLE PRECISION GRAD,GFRAC,ALBEDO,BARRIER,NMONO
      DOUBLE PRECISION SITES,HOPRATIO,DUTY,CRTEMP
      DOUBLE PRECISION TOTSITES,DENSITES
      LOGICAL MODRATES, RDCOMP

C HLOSS = rate of H destruction/H2 formation
C STICK = sticking probability for H atoms
C Ax,Bx,Cx = constants for H-atom sticking coefficients (various units)
C ECHEM = H-atom chemisorption barrier (K)
C GRAD = grain radius (cm)
C GFRAC = fractional abundance of grain (relative to H nuclei density)
C ALBEDO = grain albedo in far UV
C BARRIER = barrier between grain surface sites (A)
C NMONO = number of chemically active monolayers
C SITES = density of grain-surface sites (cm-2)
C HOPRATIO = ratio of surface hopping to desorption energies
C DUTY = duty cycle of the the grain for cosmic-ray-induced thernal desorption
C CRTEMP = maximum temperature reached by grain per cosmic-ray impact (K)
C TOTSITES = total number of surface sites per grain
C DENSITES = number density of surface sites (cm-3)  
C MODRATES = switch for adoption of modified rates
C RDCOMP = switch for inclusion of reacion-diffusion competition
            
      DOUBLE PRECISION A_RDIFF,B_RDIFF,A_DES,B_DES
      DOUBLE PRECISION A_MASS,B_MASS
      DOUBLE PRECISION A_RQUAN,B_RQUAN,KQUAN,BRANCH
      DOUBLE PRECISION A_FREQ,B_FREQ,FREQ,KAPPA

C x_RDIFF = hopping/diffusion rate (s-1)
C x_DES = binding energies (K)
C x_MASS = masses (g)
C x_QUAN = quantum tunnelling rate (s-1)
C KQUAN = quantum tunnelling reaction rate (cm3 s-1)
C BRANCH = branching ratio for reactive desorption     
C X_FREQ = vibrational frequency
C KAPPA = probability for grain-surface reaction
      
      DOUBLE PRECISION PI,HBAR,KERG,AMU,ECHARGE,C,EMASS
      DOUBLE PRECISION PLANCK,STEF,YR2SEC,AU2CM,ANG2CM

C PI = 3.14159 ...
C HBAR = Planck's reduced constant (h/2pi; erg s)
C PLANCK = Planck's constant (h erg s)
C KERG = Boltzmann's constant (erg K-1)
C AMU = atomic mass unit
C ECHARGE = electron charge
C C = speed of light (cm s-1)
C EMASS = mass of electron (amu)
C STEF = Stefan-Boltzmann constant (erg cm-2 s-1 K-4)
C YR2SEC = conversion factor from years to seconds (sec yr-1)
C AU2CM = conversion factor from AU to cm (cm AU-1)
C ANG2CM = conversion factor from Angstroms to cm (cm A-1)

       DOUBLE PRECISION DUMMY          
     
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Declaration of common blocks
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Parameters required for ODE file
                
      COMMON/BLK1/K,X,TOTAL,HLOSS,DENSITES,NMONO
     
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Data statements
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      DATA PI,AMU,KERG/3.141593,1.6735E-24,1.3806E-16/
      DATA HBAR,PLANCK /1.054571726E-27,6.62606957E-27/
      DATA EMASS /5.48579909E-04/
      DATA STEF,ECHARGE,C /5.6704E-05,4.80320425E-10,2.99792458E+10/
      DATA YR2SEC,AU2CM,ANG2CM /3.1536E7,1.49597871E13,1.0E-08/

      DATA NCON,TOTAL/1,0.00/
        
      DATA (ELEM(I),I=1,NELEM)
     *   /'He','C','N','O','Si','S','Fe','Na','Mg','Cl'/
     
      DATA (FRAC(I),I=1,NELEM)
     *   /9.75e-02,1.40e-04,7.50e-05,3.20e-04,8.00e-09,8.00e-08,
     *    3.00e-09,2.00e-09,7.00e-09,4.00e-09 /

C Common block for reaction parameters to avoid repeated file reopening
      LOGICAL INIT_REACFILE
      DATA INIT_REACFILE /.FALSE./
      COMMON /REACPARAMS/ RE1,RE2,RE3,
     *              P1,P2,P3,P4,P5,
     *              ALPHA,BETA,GAMMA,LOWTEMP,UPTEMP,
     *              RTYPE,NREAC,INIT_REACFILE

C Same for BINDFILE
      LOGICAL INIT_BINDFILE
      DATA INIT_BINDFILE /.FALSE./
      COMMON /BINDPARAMS/ SICE,BIND,YIELD,NICE,INIT_BINDFILE

C Same for RADFILE
      LOGICAL INIT_RADFILE
      DATA INIT_RADFILE /.FALSE./
      COMMON /RADPARAMS/ HABING,ZISM,CRPHOT,INIT_RADFILE

C Same for GRAINFILE
      LOGICAL INIT_GRAINFILE
      DATA INIT_GRAINFILE /.FALSE./
      COMMON /GRAINPARAMS/ GRAD,ALBEDO,BARRIER,SITES,HOPRATIO,
     *               DUTY,CRTEMP,BRANCH,MODRATES,RDCOMP,
     *               INIT_GRAINFILE

C Same for SWITCHFILE
      LOGICAL INIT_SWITCHFILE
      DATA INIT_SWITCHFILE /.FALSE./
      COMMON /SWITCHPARAMS/ NSWITCH,SWITCH,INIT_SWITCHFILE

C Flag to indicate whether to write rate coefficients to file
      LOGICAL WRITE_RATE_COEFFS

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Open and read reaction file
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      IF (.NOT. INIT_REACFILE) THEN
         OPEN(UNIT=1,FILE=REACFILE)
      
         I = 1
         NREAC = 0
      
100      READ(1,10,END=101) RE1(I),RE2(I),RE3(I),
     1      P1(I),P2(I),P3(I),P4(I),P5(I),
     2      ALPHA(I),BETA(I),GAMMA(I),LOWTEMP(I),UPTEMP(I),
     3      RTYPE(I)
          
         I = I + 1
         NREAC = NREAC + 1
            
         GO TO 100
      
101      CONTINUE

10       FORMAT(5X,8(A10),E8.2,F9.2,F10.1,2(I5),X,I2,X,1PE8.2) 

         CLOSE(UNIT=1)

         INIT_REACFILE = .TRUE.
      ENDIF

      print*,'iPOINT=',IPOINT,' NREAC=',NREAC
      print*,'INIT_REACFILE=',INIT_REACFILE
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Open and read binding energies file
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IF (.NOT. INIT_BINDFILE) THEN
         OPEN(UNIT=1,FILE=BINDFILE)

         I = 1
         NICE = 0

300      READ(1,*,END=301) SICE(I),BIND(I),DUMMY,YIELD(I)
      
         I = I + 1
         NICE = NICE + 1

         GO TO 300

301      CONTINUE

         CLOSE(UNIT=1)

         INIT_BINDFILE = .TRUE.
      ENDIF
 
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Set total number of species (include conserved species)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      NTOT = NSPEC+NCON
                      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Open and read radiation parameters
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IF (.NOT. INIT_RADFILE) THEN
         OPEN(UNIT=1,FILE=RADFILE)
      
         READ(1,*)
         READ(1,*)
         READ(1,*) GISM
         READ(1,*)
         READ(1,*) HABING
         READ(1,*)
         READ(1,*) ZISM
         READ(1,*)
         READ(1,*) CRPHOT
      
         CLOSE(UNIT=1)

         INIT_RADFILE = .TRUE.
      ENDIF

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C      Open and read grain parameters 
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IF (.NOT. INIT_GRAINFILE) THEN
         OPEN(UNIT=1,FILE=GRAINFILE)

         READ(1,*)
         READ(1,*)
         READ(1,*) GRAD
         READ(1,*)
         READ(1,*) ALBEDO
         READ(1,*)
         READ(1,*) BARRIER
         READ(1,*)
         READ(1,*) SITES
         READ(1,*)
         READ(1,*) HOPRATIO
         READ(1,*)
         READ(1,*) NMONO
         READ(1,*)
         READ(1,*) DUTY
         READ(1,*)
         READ(1,*) CRTEMP
         READ(1,*)
         READ(1,*) BRANCH
         READ(1,*)
         READ(1,*) MODRATES
         READ(1,*)
         READ(1,*) RDCOMP
    
         CLOSE(UNIT=1)

         INIT_GRAINFILE = .TRUE.
      ENDIF

C Total number of surface sites per grain (no units)  

      TOTSITES = SITES*(4*PI*GRAD**2)

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C      Open and read reaction parameters 
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IF (.NOT. INIT_SWITCHFILE) THEN
         OPEN(UNIT=1,FILE=SWITCHFILE)
         READ(1,*)
         READ(1,*)
         READ(1,*) NSWITCH
         READ(1,*)
      
         IF (NSWITCH.NE.0) READ(1,*) (SWITCH(I),I=1,NSWITCH)

         CLOSE(UNIT=1)

         INIT_SWITCHFILE = .TRUE.
      ENDIF

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     UV photon flux
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C UV photon flux i.e. in units of photons cm-2 s-1
     
C Scale UV photon flux by the Habing field (ISRF)

      UVPHOT = HABING*EXP(-AV*3.02)

C Add on internally generated UV photons

      UVPHOT = UVPHOT + CRPHOT*(ZETACR/ZISM) 

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Estimate column densities for self-shielding
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
            
      N_H2 = 0.5*1.87E+21*AV
      N_CO = 8e-4*N_H2
      N_N2 = 4e-5*N_H2

      ! write(*,*) N_CO, N_N2

      DO I=1,NSPEC
          IF(SPEC(I).EQ.'CO') N_CO = Y(I)/DENS*N_H2
          
      !     IF(SPEC(I).EQ.'CO') write(*,*) Y(I),DENS,N_H2,N_CO

          IF(SPEC(I).EQ.'N2') N_N2 = Y(I)/DENS*N_H2
      END DO
      ! write(*,*) N_CO, N_N2
                                              
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCC   BEGIN INITIALISING MODEL   CCCCCCCCCCCCCCCCCCCCCCCC    
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Add conserved species to species list and assign initial abundance
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Set grain fractional abundance (used in rates)

      GFRAC = 0.0
      
      DO I=1,NSPEC
         IF(SPEC(I).EQ.'GRAIN0') GFRAC = GFRAC + Y(I)      
         IF(SPEC(I).EQ.'GRAIN-') GFRAC = GFRAC + Y(I)      
      END DO
      
      GFRAC = GFRAC/DENS

C Add on electrons to total number of species
      
      DO I=1,NCON
         X(I) = 0.0
     	   SPEC(NSPEC+I)='e-'         
      END DO	

      TOTAL = X(1)*DENS

C Number density of grain-surface sites (cm-3)  

      DENSITES = TOTSITES*GFRAC*DENS 
           
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Calculating rate coefficients
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                 
      DO I=1,NREAC

C Reaction type 1: default two-body rate coefficient  (cm3 s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     
      IF(RTYPE(I).EQ.1) THEN
      
      IF ((TEMPGAS.GE.LOWTEMP(I)).AND.(TEMPGAS.LT.UPTEMP(I))) THEN              
         K(I) = ALPHA(I)*((TEMPGAS/300.0)**BETA(I))
     *      *EXP(-GAMMA(I)/TEMPGAS)   
      ELSE
         K(I)=0.0
      END IF
      
      END IF

C Reaction type 2: direct cosmic-ray ionisation  (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
 
      IF(RTYPE(I).EQ.2) THEN  
         
         K(I) = ALPHA(I)*((ZETACR+ZETAXR)/ZISM) 
                     
      END IF	

C Reaction type 3: cosmic-ray-induced photoreaction (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      IF(RTYPE(I).EQ.3) THEN
         K(I)= ALPHA(I)*((ZETAXR+ZETACR)/ZISM)
     *      *((TEMPGAS/300.0)**BETA(I))
     *      *GAMMA(I)/(1.0-ALBEDO)
      END IF	

C Reaction type 4: photoreaction (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      IF(RTYPE(I).EQ.4) THEN

         K(I) = ALPHA(I)*EXP(-GAMMA(I)*AV) 

C Self-shielding of H2

         IF(RE1(I).EQ.'H2') THEN
      
         CALL SHIELDING(RE1(I),N_H2,N_CO,N_N2,TEMPGAS,SHIELD) 
      
         K(I) = K(I)*SHIELD
      
C Self- and mutual-shielding of CO

         ELSE IF (RE1(I).EQ.'CO') THEN
      !    write(*,*) 'SHIELD_IN', N_H2, N_CO, N_N2, TEMPGAS
      
         CALL SHIELDING(RE1(I),N_H2,N_CO,N_N2,TEMPGAS,SHIELD)

         K(I) = K(I)*SHIELD
      !    write(*,*) K(I)
      
C Self- and mutual-shielding of N2

         ELSE IF (RE1(I).EQ.'N2') THEN
      
         CALL SHIELDING(RE1(I),N_H2,N_CO,N_N2,TEMPGAS,SHIELD)

         K(I) = K(I)*SHIELD
      
         END IF
         
      END IF

C Reaction type 5: direct X-ray ionisation (s-1) 
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C Not needed for protostellar/cloud models
     
C Reaction type 6: grain-cation recombination rate (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = branching ratio
C GAMMA = mass (amu)

C For X+ + G- ---> X0 + G0 
C Enhancement in rate according to Draine & Sutin (1987)

C K = Racc * { 1 + e^2/(akT) } * { 1 + sqrt(2e^2/(akT + 2e^2)) }

      IF(RTYPE(I).EQ.6) THEN
         K(I) = ALPHA(I)*(PI*GRAD**2.0)*GFRAC*DENS
     1      *SQRT(8.0*KERG*TEMPGAS/(PI*AMU*GAMMA(I)))
     2      *(1.0 + (ECHARGE**2.0/(GRAD*KERG*TEMPGAS)))
     3      *(1.0 + SQRT(2.0*ECHARGE**2.0
     4      /((GRAD*KERG*TEMPGAS) + 2.0*ECHARGE**2.0))) 
      END IF
      
C Reaction type 7: neutral grain accretion rate (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = branching ratio
C GAMMA = mass (amu)
     
      IF(RTYPE(I).EQ.7) THEN
         K(I) = ALPHA(I)*(PI*GRAD**2.0)*GFRAC*DENS
     *      *SQRT(8.0*KERG*TEMPGAS/(PI*AMU*GAMMA(I)))
      END IF

C Reaction type 8: thermal desorption rate (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = binding energy (K)
C GAMMA = mass (g)

      IF(RTYPE(I).EQ.8) THEN

C Look up binding energy

         DO J=1,NICE              
            IF(SICE(J).EQ.RE1(I)) ALPHA(I) = BIND(J)
         END DO
         
         K(I)= SQRT((2*SITES*KERG*ALPHA(I))
     1      /((PI**2)*AMU*GAMMA(I)))
     2      *NMONO*DENSITES
     3      *EXP(-ALPHA(I)/TEMPDUST)
                            
      END IF	

C Reaction type 9: cosmic-ray-induced thermal desorption rate (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = binding energy (K)
C GAMMA = mass (g)
C DUTY = duty cycle of the grains
C CRTEMP = 70 K

      IF(RTYPE(I).EQ.9) THEN

C Look up binding energy

         DO J=1,NICE              
            IF(SICE(J).EQ.RE1(I)) ALPHA(I) = BIND(J)
         END DO
           
         K(I)= (ZETACR/ZISM)*DUTY
     1      *SQRT((2*SITES*KERG*ALPHA(I))
     2      /((PI**2)*AMU*GAMMA(I)))
     3      *NMONO*DENSITES
     4      *EXP(-ALPHA(I)/CRTEMP)
               
      END IF

C Reaction type 10: photodesorption rate (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = photodesorption yield (photon-1)

      IF(RTYPE(I).EQ.10) THEN

C Look up photodesorption yield

         DO J=1,NICE              
            IF(SICE(J).EQ.RE1(I)) ALPHA(I) = YIELD(J)
         END DO
          
         K(I) = UVPHOT*ALPHA(I)*
     *      NMONO*(4.0*PI*GRAD**2.0)*GFRAC*DENS
         
      END IF

C Reaction type 11: grain-surface cosmic-ray-induced photoreaction rate (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C Use the same as the gas-phase rate

      IF(RTYPE(I).EQ.11) THEN
      
         K(I)= ALPHA(I)*((ZETAXR+ZETACR)/ZISM)
     *      *((TEMPGAS/300.0)**BETA(I))
     *      *GAMMA(I)/(1.0-ALBEDO)
     
      END IF

C Reaction type 12: grain-surface photoreaction rate (s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C Use the same as the gas-phase rate ...

      IF(RTYPE(I).EQ.12) THEN
      
         K(I) = ALPHA(I)*EXP(-GAMMA(I)*AV) 

C Self-shielding of H2

         IF(RE1(I).EQ.'GH2') THEN
      
         CALL SHIELDING(RE1(I),N_H2,N_CO,N_N2,TEMPGAS,SHIELD) 
      
         K(I) = K(I)*SHIELD
      
C Self- and mutual-shielding of CO

         ELSE IF (RE1(I).EQ.'GCO') THEN
      
         CALL SHIELDING(RE1(I),N_H2,N_CO,N_N2,TEMPGAS,SHIELD)

         K(I) = K(I)*SHIELD
      
C Self- and mutual-shielding of N2

         ELSE IF (RE1(I).EQ.'GN2') THEN
      
         CALL SHIELDING(RE1(I),N_H2,N_CO,N_N2,TEMPGAS,SHIELD)

         K(I) = K(I)*SHIELD
      
         END IF

       END IF

C Reaction type 13: grain-surface two-body reaction rate (cm3 s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = reaction barrier (K)
C BETA = branching ratio 

C Look up binding energies and masses

      IF((RTYPE(I).EQ.13).OR.(RTYPE(I).EQ.14)) THEN

         A_RDIFF = 0.0
         B_RDIFF = 0.0
             
C Select for first reactant

         DO J=1,NICE           
            IF (RE1(I).EQ.SICE(J)) A_DES = BIND(J)
         END DO
             
         DO J=1,NSPEC           
            IF (RE1(I).EQ.SPEC(J)) A_MASS = MASS(J)
         END DO
         
         IF(A_DES.EQ.0.0) STOP

C Calculate classical diffusion/hopping rate for first reactant
             
         A_FREQ = SQRT((2*SITES*KERG*A_DES)/((PI**2)*AMU*A_MASS))
         
         A_RDIFF = A_FREQ*EXP(-(A_DES*HOPRATIO)/TEMPDUST)/TOTSITES

         IF ((RE1(I).EQ.'GH').OR.(RE1(I).EQ.'GH2')) THEN

C Calculate quantum diffusion rate (for H and H2 only)

            A_RQUAN = A_FREQ*EXP(-2*(BARRIER/HBAR)
     1         *SQRT(2*A_MASS*AMU*HOPRATIO*A_DES*KERG))/TOTSITES

C Select quantum tunnelling if faster than classical diffusion/hopping

            IF(A_RQUAN.GT.A_RDIFF) A_RDIFF = A_RQUAN
                
         END IF

C Set vibrational frequency for reaction-diffusion competition

         FREQ = A_FREQ
             
C Select for second reactant
                           
         DO J=1,NICE           
            IF (RE2(I).EQ.SICE(J)) B_DES = BIND(J)
         END DO
             
         DO J=1,NSPEC           
            IF (RE2(I).EQ.SPEC(J)) B_MASS = MASS(J)
         END DO

         IF(B_DES.EQ.0.0) STOP

C Calculate classical diffusion/hopping rate for second reactant
                          
         B_FREQ = SQRT((2*SITES*KERG*B_DES)/((PI**2)*AMU*B_MASS))
         
         B_RDIFF = B_FREQ*EXP(-(B_DES*HOPRATIO)/TEMPDUST)/TOTSITES
         
         IF ((RE2(I).EQ.'GH').OR.(RE2(I).EQ.'GH2')) THEN

C Calculate quantum diffusion rate (for H and H2 only)

            B_RQUAN = B_FREQ*EXP(-2*(BARRIER/HBAR)
     1         *SQRT(2*B_MASS*AMU*HOPRATIO*B_DES*KERG))/TOTSITES

C Select quantum tunnelling if faster than classical diffusion/hopping

            IF(B_RQUAN.GT.B_RDIFF) B_RDIFF = B_RQUAN
          
         END IF

C Choose largest vibrational frequency for reaction-diffusion competition

         IF (B_FREQ.GT.FREQ) FREQ = B_FREQ
                                              
C Calculate classical reaction rate probability

         KAPPA = EXP(-ALPHA(I)/TEMPDUST)
                   
         IF((RE1(I).EQ.'GH').OR.(RE1(I).EQ.'GH2').OR.
     *      (RE2(I).EQ.'GH').OR.(RE2(I).EQ.'GH2')) THEN

C Calculation quantum tunnelling reaction probability (for H and H2 only)
         
         KQUAN = EXP(-2*(BARRIER/HBAR)*SQRT((2*AMU*KERG)
     1      *((A_MASS*B_MASS)/(A_MASS+B_MASS))*ALPHA(I)))
    
C Select quantum tunnelling rate if faster than classical reaction rate
              
         IF(KQUAN.GT.KAPPA) KAPPA = KQUAN
    
         END IF
         
C Calculate probability of reaction (reaction-diffusion competition)
C See e.g., Garrod & Pauly 2011 (equation 6)
C Switch off until fully tested

C      KAPPA=FREQ*KAPPA/(FREQ*KAPPA+A_RDIFF+B_RDIFF)    

C Calculate reaction rate
      
      K(I) = KAPPA*(A_RDIFF+B_RDIFF)*(NMONO**2)*(DENSITES**2)/
     *   (GFRAC*DENS)
               
      END IF             
                        
C Reaction type 14: reactive desorption (cm3 s-1)  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C BRANCH = branching ratio for reactive desorption

      IF(RTYPE(I).EQ.14) K(I) = K(I)*BRANCH

C  Reaction type 15: Three-body reaction rates (cm3 s-1)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = pre-exponential factor (cm6 s-1)
C BETA = temperature exponent
C GAMMA =  reaction barrier (K)
C Multiply by number density for quasi-two-body rate (cm3 s-1)

      IF(RTYPE(I).EQ.15) THEN

      IF ((TEMPGAS.GE.LOWTEMP(I)).AND.(TEMPGAS.LT.UPTEMP(I))) THEN              
          K(I) = ALPHA(I)*((TEMPGAS/300.0)**BETA(I))
     *       *EXP(-GAMMA(I)/TEMPGAS)*DENS
      END IF
         
      END IF

C  Reaction type 16: collisional dissociation (s-1)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = pre-exponential factor (cm3 s-1)
C BETA = temperature exponent
C GAMMA =  reaction barrier (K)

      IF(RTYPE(I).EQ.16) THEN

          K(I) = ALPHA(I)*((TEMPGAS/300.0)**BETA(I))
     *       *EXP(-GAMMA(I)/TEMPGAS)*DENS

      END IF

C  Reaction type 17: collisional de-excitation of H2*
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C ALPHA = pre-exponential factor (cm3 s-1)
C BETA = temperature exponent
C GAMMA =  reaction barrier (K)

      IF(RTYPE(I).EQ.17) THEN
          
          K(I) = ALPHA(I)*((TEMPGAS/300.0)**BETA(I))*
     *       EXP(-GAMMA(I)/(TEMPGAS+1200.0))  

      END IF

C  Reaction type 18: Lyman-alpha photorate
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Not used in envelope model

C ALPHA = Lyman-alpha cross section (cm-2)
C SCALE = branching ratio

      IF(RTYPE(I).EQ.18) THEN

         K(I) = 0.0         

      END IF 

C  Reaction type 19: Radiative de-excitation of H2*
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      IF(RTYPE(I).EQ.19) THEN

         K(I) = ALPHA(I)         

      END IF 

C  Reaction type 20: Grain electron capture rate
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     
      IF(RTYPE(I).EQ.20) THEN

         K(I) = (PI*GRAD**2.0)*SQRT(8.0*KERG*TEMPGAS/(PI*AMU*EMASS))

      END IF 

C SWITCH OFF REACTIONS HERE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      IF (NSWITCH.NE.0) THEN
         DO J=1,NSWITCH
            IF(RTYPE(I).EQ.SWITCH(J)) K(I) = 0.0
         END DO
      END IF   
                                              
C END OF REACTION RATE COEFFICIENTS
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Write reactions and rate coefficients to file

      IF(K(I).LT.1D-50) K(I) = 0.0

C Only write if unit 10 has been opened
      INQUIRE(UNIT=10,OPENED=WRITE_RATE_COEFFS)
      IF (WRITE_RATE_COEFFS) THEN
         WRITE(10,'(I4,X,7(A10),1PE15.5)') I,RE1(I),RE2(I),RE3(I),
     *      P1(I),P2(I),P3(I),P4(I),K(I)
      ENDIF
 
      END DO
                                                
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     HLOSS term to model H --> H2 formation on grain surface
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C Use temperature-dependent sticking coefficients from Sha et al. 2005 
C and Cuppen et al. 2010
C Need to consider both physisorption and chemisorption

C S_{ph} = (1 + b1*sqrt(Tgas + Tgrain) + b2*Tgas - b3*Tgas**2.0)**-1
C b1 = 4.2 x 10^{-2} K-0.5
C b2 = 2.3 x 10^{-3} K-1
C b3 = 1.3 x 10^{-7} K-2

C S_{ch} = exp{-Ech/Tgas}*(1 + c1*sqrt(Tgas + Tgrain) + c2*Tgas**4.0)**-1
C Ech = 0.15 eV = 1741 K 
C c1 = 5 x 10^{-2} K-0.5
C c2 = 1 x 10^{-14} K-4
      
      A1 = 4.2E-02
      A2 = 2.3E-03
      A3 = 1.3E-07
      
      STICK = (1 + A1*SQRT(TEMPGAS+TEMPDUST) + 
     *   A2*TEMPGAS - A3*(TEMPGAS**2.0))**(-1.0)
      
      ECHEM = 1741.0
      B1 = 5.0E-02
      B2 = 1.0E-14
      
      STICK = STICK + EXP(-ECHEM/TEMPGAS)/
     *   (1 + B1*SQRT(TEMPGAS+TEMPDUST) + B2*TEMPGAS**4.0)
      
      HLOSS = STICK*(PI*GRAD**2.0)*GFRAC*DENS
     *      *SQRT(8.0*KERG*TEMPGAS/(PI*AMU))

C Turn off explicit H2 formation if HLOSS is switched on

      IF(HLOSS.NE.0.0) THEN
     
      DO I=1,NREAC      
        IF((RE1(I).EQ.'GH').AND.(RE2(I).EQ.'GH')) THEN 
           K(I) = 0.0
        END IF         
      END DO
     
      END IF
                        
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCC     BEGIN INTEGRATION    CCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Call DVODE integrator
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C Set time counter to zero      
      T0 = 0.0
             
20    CONTINUE

      CALL DRIVE(NSPEC,T0,Y,TIME,DENS)
                            
C Output abundances relative to total H nuclei density
          
      DO I=1,NSPEC
         ABUN(I) = Y(I)/DENS
      END DO

      DO I=1,NCON
      	ABUN(NSPEC+I) = X(I)/DENS
      END DO    
      
C Call Analyse subroutine for a particular grid point

      IF((ANA .EQV. .TRUE.).AND.(IPOINT.EQ.IANA)) THEN 
         
         WRITE(*,*)
         WRITE(*,*) 'Calling analyse subroutine ...'

         CALL ANALYSE(NSPEC,SPEC,NREAC,Y,X,K,TTOT,RE1,RE2,RE3,
     *      P1,P2,P3,P4,ANAFILE)

         WRITE(*,*) '... done!'
         WRITE(*,*)
         
      END IF   
                          
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCC     RETURN    CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      RETURN
      
      END SUBROUTINE MAIN