
       PROGRAM Chemistry

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Declaration of variables
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       
       IMPLICIT NONE
       
       INTEGER I,J,K

C I,J,K = counters       
              
       INTEGER D,E,F,G              
       PARAMETER (D=1600)
       PARAMETER (F=100)
       PARAMETER (G=1000)

C D = max number of grid points
C F = max number of time steps
C G = max number of species

       INTEGER NGRID

C NGRID = number of disk model grid points
       
       CHARACTER*70 INPFILE,PHYSFILE
       CHARACTER*70 REACFILE,SPECFILE,BINDFILE
       CHARACTER*70 GRAINFILE,RADFILE,SWITCHFILE
       CHARACTER*70 OUTFILE,RATESFILE,ANAFILE
       
       LOGICAL ANA

C INPFILE = master file containing filenames 
C PHYSFILE = disk physical input filename
C REACFILE = reaction rates filename
C SPECFILE = species filename
C BINDFILE = binding energy filename
C SWITCHFILE = reaction parameters filename
C GRAINFILE = grain parameters filename
C RADFILE = radiation field parameters filename
C OUTFILE = output filename
C RATESFILE = rates filename
C ANAFILE = analyse output filename
       
       INTEGER IANA
       DOUBLE PRECISION XCOORD(D),YCOORD(D),ZCOORD(D)
       DOUBLE PRECISION DENS(D),TEMPGAS(D),TEMPDUST(D)
       DOUBLE PRECISION AV(D),ZETACR(D),ZETAXR(D)
       DOUBLE PRECISION TIME(D),DELTA(D)

C IANA = grid point at which to call analyse subroutine
C XCOORD, YCOORD, ZCOORD = model coordinates 
C DENS = H nuclei number density (cm-3)
C TEMPGAS = disk gas temperature (K)
C TEMPDUST = disk dust temperature (K)
C GFUV = disk wavelength-integrated UV field (erg cm-2 s-1)  
C ZETACR = disk cosmic-ray ionisation rate (s-1)
C ZETAXR = disk x-ray ionisation rates (s-1)
C TIME = time evolution (s)
C DELTA = delta time (timestep) (s)
C AV = visual extiction (mag)

       DOUBLE PRECISION GISM   
       
C GISM = interstellar radiation field (erg cm-2 s-1)

       INTEGER NSPEC,NTOT
       DOUBLE PRECISION ABUN(G),Y(G),MASS(G)
       CHARACTER*10 SPEC(G)

C NSPEC = number of species
C NTOT = total number of species
C ABUN = radial/time-dependent abundances (relative to H nuclei density)
C Y = species abundances (cm-3)
C MASS = species mass (amu)
C SPEC = species names               
       
       DOUBLE PRECISION DUMDP,YR2SEC
       CHARACTER*10 DUMCH
       
       DATA YR2SEC/3.1536e+07/
        
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC       
                           
       WRITE(*,*)
       WRITE(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
       WRITE(*,*) '+++++        ENVELOPE CHEMICAL MODEL         ++++++'
       WRITE(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
       WRITE(*,*)
       WRITE(*,*) 'Version 1.0 authored by Catherine Walsh and Maria '     
       WRITE(*,*) '   Drozdovskaya, Leiden Observatory (01/08/2013)'       
       WRITE(*,*)
       WRITE(*,*) 'Version 1.1 amended by Catherine Walsh (18/05/2014)'       
       WRITE(*,*)
       WRITE(*,*) 'Version 1.2 amended by Catherine Walsh (13/08/2015)'       
       WRITE(*,*)
       WRITE(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
       WRITE(*,*)
              
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Initialise model parameters
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

       NGRID = 0              

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Read master input file from standard input
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
              
      CALL GETARG(1,INPFILE)
      
      INPFILE = TRIM(INPFILE)
      
      WRITE(*,'(X,A22,X,A70)') 'Input file = ', INPFILE
            
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Open and read file containing filenames
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
            
      OPEN(UNIT=1,FILE=INPFILE)
      
      READ(1,*)
      READ(1,*)
      READ(1,'(A70)') PHYSFILE
      READ(1,*)
      READ(1,'(A70)') REACFILE
      READ(1,*)
      READ(1,'(A70)') SPECFILE
      READ(1,*)
      READ(1,'(A70)') BINDFILE
      READ(1,*)
      READ(1,'(A70)') SWITCHFILE
      READ(1,*)
      READ(1,'(A70)') GRAINFILE
      READ(1,*)
      READ(1,'(A70)') RADFILE
      READ(1,*)
      READ(1,'(A70)') OUTFILE
      READ(1,*)
      READ(1,'(A70)') RATESFILE

      READ(1,*)
      READ(1,*) DUMCH
            
      IF((DUMCH(1:1).EQ.'T').OR.(DUMCH(1:1).EQ.'t')) ANA = .TRUE.

      IF(ANA .EQV. .TRUE.) THEN
         READ(1,*)
         READ(1,*) IANA
         READ(1,*)
         READ(1,'(A70)') ANAFILE
      END IF   

      CLOSE(UNIT=1)
      
      PHYSFILE   = TRIM(PHYSFILE)
      REACFILE   = TRIM(REACFILE)
      SPECFILE   = TRIM(SPECFILE)
      BINDFILE   = TRIM(BINDFILE)
      GRAINFILE  = TRIM(GRAINFILE)
      SWITCHFILE = TRIM(SWITCHFILE)
      RADFILE    = TRIM(RADFILE)      
      
      OUTFILE    = TRIM(OUTFILE)      
      RATESFILE  = TRIM(RATESFILE)      

      WRITE(*,*)
      WRITE(*,'(X,A22,X,A70)') 'Physical conditions  =', PHYSFILE
      WRITE(*,'(X,A22,X,A70)') 'Chemical network     =', REACFILE
      WRITE(*,'(X,A22,X,A70)') 'Species file         =', SPECFILE
      WRITE(*,'(X,A22,X,A70)') 'Binding energies     =', BINDFILE
      WRITE(*,'(X,A22,X,A70)') 'Reaction parameters  =', SWITCHFILE
      WRITE(*,'(X,A22,X,A70)') 'Grain parameters     =', GRAINFILE
      WRITE(*,'(X,A22,X,A70)') 'Radiation parameters =', RADFILE
      WRITE(*,*)      
      WRITE(*,'(X,A22,X,A70)') 'Output file          =', OUTFILE
      WRITE(*,'(X,A22,X,A70)') 'Rates file           =', RATESFILE
      
      IF(ANA .EQV. .TRUE.) THEN
         ANAFILE  = TRIM(ANAFILE)            
         WRITE(*,'(X,A22,X,A70)') 'Analyse file         =', ANAFILE
      END IF
      
      WRITE(*,*)      
           
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Open and read physical conditions file
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      WRITE(*,*) 'Opening and reading physical conditions file ...'
      WRITE(*,*)
      
      OPEN(UNIT=1,FILE=PHYSFILE)
      
      I = 1
      NGRID = 0
      
100   READ(1,*,END=101) XCOORD(I),YCOORD(I),ZCOORD(I),DENS(I),
     *   TEMPGAS(I),TEMPDUST(I),AV(I),ZETAXR(I),ZETACR(I),TIME(I)
                                               
      I = I + 1
      NGRID = NGRID + 1
      
      GO TO 100

101   CONTINUE

      CLOSE(UNIT=1)
            
C Set timesteps for integration

      DELTA(1) = TIME(1)
            
      DO I=2,NGRID      
         DELTA(I) = TIME(I) - TIME(I-1)
      END DO

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Open and read initial abundances file
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      OPEN(UNIT=1,FILE=SPECFILE)
      
      I = 1
      NSPEC = 0

102   READ(1,*,END=103) DUMDP,SPEC(I),ABUN(I),MASS(I)
      
      I = I + 1
      NSPEC = NSPEC + 1
      
      GO TO 102
      
103   CONTINUE

      CLOSE(UNIT=1)
                                                           
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C      Solve chemistry for each grid point
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       
      WRITE(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
      WRITE(*,*) 'Beginning to run model ...'
      WRITE(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
      WRITE(*,*)
      WRITE(*,*) 'Preparing output files ...'
      WRITE(*,*)
      
      OPEN(UNIT=9, FILE=OUTFILE)
      OPEN(UNIT=10,FILE=RATESFILE)
      
      WRITE(10,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
      WRITE(10,*) '+++++++++   Reaction Rate Coefficients   ++++++++++'
      WRITE(10,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
      
      WRITE(9,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'       
      WRITE(9,*) '++++++        ENVELOPE CHEMICAL MODEL         +++++'
      WRITE(9,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
      WRITE(9,*)

      WRITE(*,*) 'Progress = ...'

      DO 200 I = 1, NGRID

      WRITE(*,'(X,A4,I4,A2,I4)') '... ', I, ' /', NGRID
      WRITE(10,'(X,A4,I4,A2,I4)') '... ', I, ' /', NGRID

      IF (.NOT. (ANA .AND. (I .GT. IANA))) THEN

C Convert fractional abundances to absolute abundances
      
      DO J=1,NSPEC
         Y(J) = ABUN(J)*DENS(I)
      END DO

C Call chemistry subroutine
       
       CALL MAIN(DENS(I),TEMPGAS(I),TEMPDUST(I),AV(I),
     1   ZETACR(I),ZETAXR(I),DELTA(I),I,TIME(I),ANA,IANA,ANAFILE,
     2   REACFILE,SPECFILE,BINDFILE,GRAINFILE,RADFILE,SWITCHFILE,
     3   GISM,MASS,Y,ABUN,SPEC,NSPEC,NTOT)

      END IF


C Write preamble for grid point in output file
      

         WRITE(9,'(X,A10,1PE10.3,X,A2)')  'X COORD = ',
     *      XCOORD(I), 'PC'
         WRITE(9,'(X,A10,1PE10.3,X,A2)')  'Y COORD = ',
     *      YCOORD(I), 'PC'
         WRITE(9,'(X,A10,1PE10.3,X,A2)')  'Z COORD = ',
     *      ZCOORD(I), 'PC'
         WRITE(9,'(X,A10,1PE10.3,X,A4)') 'DENSITY = ',
     *      DENS(I), 'CM-3' 
         WRITE(9,'(X,A18,1PE10.3,X,A)') 'GAS TEMPERATURE = ',
     *      TEMPGAS(I), 'K' 
         WRITE(9,'(X,A19,1PE10.3,X,A)') 'DUST TEMPERATURE = ',
     *      TEMPDUST(I), 'K' 
         WRITE(9,'(X,A19,1PE10.3,X,A3)')  'VISUAL EXTINCTION = ',
     *      AV(I),'MAG'
         WRITE(9,*)

         WRITE(10,*)
       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Output Routine
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
              
	 WRITE(9,90) TIME(I)/YR2SEC
         WRITE(9,*)'+++++++++++++++++++++++++++++++++++++++++++++++++++'
          
	 DO K=1,NTOT
	    WRITE(9,91) SPEC(K), ABUN(K)  
	 END DO
         WRITE(9,*)
         WRITE(9,*)

90    FORMAT (/,1X,'TIME',4X,1PE12.3, ' YEARS')
91    FORMAT ((1X,A10),1PE10.3)
                    
c       END DO
200    CONTINUE

300    CONTINUE
                                   
      CLOSE(UNIT=9)
      CLOSE(UNIT=10)
                            
      WRITE(*,*)
      WRITE(*,*) '   ... DONE!'
      WRITE(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++'
      WRITE(*,*)
        
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     END OF PROGRAM
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

       END PROGRAM Chemistry
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
