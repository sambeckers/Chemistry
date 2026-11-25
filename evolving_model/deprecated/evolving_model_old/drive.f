CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C SUBROUTINE DRIVE - INTERFACE TO DVODE PACKAGE C             
CCCcCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC 

      SUBROUTINE DRIVE (NEQ,T,Y,TOUT,DENS)
      DOUBLE PRECISION T,TOUT
      DOUBLE PRECISION Y(NEQ)
      DOUBLE PRECISION RTOL,ATOL
      DOUBLE PRECISION RWORK(2000000)
      CHARACTER*11 JAC
      INTEGER NEQ,MF,INDEX,POW
      INTEGER ITOL,ITASK,ISTATE,IOPT,LRW,LIW

C IWORK is an integer work array of least
C 30 (MF=10) or 30+NEQ (MF=21,22,24,25)

      INTEGER IWORK(NEQ+30)

C CALL FILE CONTAINING ODES HERE..
      EXTERNAL DIFFUN

      INDEX  = 0

C ITOL = 1 or 2 according as ATOL (scalar or array)

      ITOL   = 1

c RTOL = relative tolerance parameter

      RTOL   = 1.0E-05

C ATOL = absolute tolerance parameter - estimated local error in Y(i)
C     if ITOL = 1.... EWT(i)=RTOL*abs(Y(i)) + ATOL
C     if ITOL = 2.....EWT(i)=RTOL*abs(Y(i)) + ATOL(i)
C     local error test passes if in each component either abs error is less
C     than ATOL or rel. error is less than RTOL
      
C Scale absolute tolerance ATOL by density (relax tolerances in case of high density)
      
C      POW = INT(LOG10(DENS))      
C      POW = 20-POW      
C      ATOL = 10**(-FLOAT(POW))

C Set absolute value for the tolerance

      ATOL = 1.0E-25
      
C ITASK = set to 1 for output every TOUT

      ITASK  = 1

C ISTATE = set to 1

      ISTATE = 1

C IOPT = 0 to indicate no optional input provided

      IOPT   = 0

C LRW = declared length of RWORK (declared above)

      LRW    = 2000000

C RWORK is a real work array and it's length must be at least;
C 20+16*NEQ  .....................(MF=10)
C 22+9*NEQ+2*NEQ**2..............(MF=21 or 22)
C 22+11*NEQ + (3*ML + "*MU)*NEQ...(MF=24 or 25)

C LIW = declared length of IWORK declared above

      LIW    = NEQ+30

C name of subroutine for jacobian matrix (MF 21,24) or pass a dummy
C name if not needed

      JAC    = 'DUMMYMATRIX'

C MF = method flag
C     10 = nonstiff, no jacobian matrix
C     21 = stiff, user supplied jacobian matrix
C     22 = stiff, internally generated full jacobian
C     24 = stiff, user supplied banded jacobian matrix
C     25 = stiff, internally generated banded jacobian

      MF     = 22

C  CALL THE DVODE SUBROUTINE
      
      CALL DVODE (DIFFUN,NEQ,Y,T,TOUT,ITOL,RTOL,ATOL,ITASK,
     &            ISTATE,IOPT,RWORK,LRW,IWORK,LIW,JAC,MF,RPAR,IPAR)

      WRITE(*,60) IWORK(11),IWORK(12),IWORK(13),IWORK(19),
     1            IWORK(20),IWORK(21),IWORK(22), RWORK(11),IWORK(16)
  60  FORMAT(/' No. steps =',I8,'   No. f-s =',I8,
     1       '   No. J-s =',I8,'   No. LU-s =',I8/
     2       '  No. nonlinear iterations =',I8/
     3       '  No. nonlinear convergence failures =',I8/
     4       '  No. error test failures =',I8/
     5       '  Last step size used =',E15.5/
     6       '  Index compenent largest error = 'I8/)


      IF(ISTATE.LT.0) THEN
         ISTATE = 1
      ENDIF 


      RETURN
      END
