CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION TO CALCULATE GSTAR, THE RATIO OF THE STELLAR UV FLUX      C
C  AT 50 STELLAR RADII TO THE DRAINE FLUX                             C
C  RATIO OF INTEGRATED FLUXES                                         C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      DOUBLE PRECISION FUNCTION GETGSTAR(T,RADIUS)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      double precision LAMDA, RADIUS
      double precision T
      CHARACTER*10 STAR,FIELD
      CHARACTER*6 MOLECULE
      CHARACTER*3 PROCESS
      CHARACTER*20 PPFILE
C T = EFFECTIVE TEMPERATURE OF STAR
      dimension LAMDA(29999),FLUX(29999)
      DATA C/2.9979E10/,pie/3.14159/
C
C IFLAG = 1 -- DRAINE INSTELLAR RADIATION FIELD
C IFLAG = 0 -- BLACKBODY RADIATION FIELD AT TEMPERATURE T
C      IFLAG = 0
C
C FOR BLACKBODY CALCULATION SET RADIUS IN UNITS OF STELLAR RADIUS
C      RADIUS = 50.0
c
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Open and read species file
C L=Line, C=Continuum
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       FIELD = 'DRAINE'
       STAR = 'BB'
       
            
C
C Open output file to write unshielded photorates 
c
      OPEN(UNIT=3,FILE='inner_unshielded_ratio_new_'//TRIM(STAR)//
     * '.out') 
      
C      IF (IFLAG.EQ.1) WRITE (3,998)
      WRITE(3,999) RADIUS
C      IF (IFLAG.NE.1) WRITE(3,298) TRIM(STAR)
      WRITE (3,296)
      
C      do 777 T = 2300,4000,10
c
c 
      SUMDRAINE = 0.0
      DO 2 I=1,1139
      LAMDA(I) = 911.0 + I

C INTEGRATED DRAINE INTERSTELLAR UV FIELD FLUX 
C
        IF (LAMDA(I).LT.2000.0)  THEN
           FLUX(I) = 3.2028E+15*(LAMDA(I)**(-3.0))
     *    -5.1542E+18*(LAMDA(I)**(-4.0))+ 2.0546E+21*(LAMDA(I)**(-5.0)) 
        END IF
        IF (LAMDA(I).GE.2000.0) THEN
           FLUX(I) = 732.0*(LAMDA(I)**0.7)
        END IF
 2    CONTINUE

      lamda(1) = 2000.0
      fluxupper = -0.5*(3.2028E+15*(LAMDA(1)**(-2.0))) +(1./3.)*
     *   (5.1542E+18*(LAMDA(1)**(-3.0))) -
     *   0.25*(2.0546E+21*(LAMDA(1)**(-4.0))) 
      lamda(2) = 912.0
      fluxlower = -0.5*(3.2028E+15*(LAMDA(2)**(-2.0))) +(1./3.)*
     *   (5.1542E+18*(LAMDA(2)**(-3.0))) -
     *   0.25*(2.0546E+21*(LAMDA(2)**(-4.0))) 
      sumdraine1 = (fluxupper - fluxlower)
      lamda(1) = 2050.0
      fluxupper = (732.0/1.7)*(lamda(1)**1.7)
      lamda(2) = 2000.0
      fluxlower = (732.0/1.7)*(lamda(2)**1.7)
      sumdraine2 = fluxupper - fluxlower
      sumdrainetot = sumdraine1 + sumdraine2
!       write(*,*) 'sumdraine1 = ',sumdraine1,'sumdraine2 = ',sumdraine2
!       write(*,*) 'integd draine field - full integral = ',sumdrainetot

C PHYSICAL CONSTANTS IN SI UNITS
C CALCULATE BAND RADIANCE FROM BB AT TEMPERATURE T
C FORMULAE BELOW FOR BAND RADIANCE IN PHOTONS PER UNIT AREA PER SECOND PER STERADIAN 
C   TAKEN FROM DOCUMENT ON SPECTRAL CALCULATOR WEBSITE
      NITER = 5
      CSI = C/100.0
      BOLTZ = 1.3807E-23
      HPLANCK = 6.6261E-34
C SET WAVELENGTHS IN CM-1 from angstroms
      SIGMA1 = 1.0e8/912.0
      SIGMA2 = 1.0e8/2050.0
C SET X1 AND X2
      X1 = 100.0*(HPLANCK*CSI*SIGMA1)/(BOLTZ*T)
      X2 = 100.0*(HPLANCK*CSI*SIGMA2)/(BOLTZ*T) 
C NOW DO SUM 
      SUM1 = 0.0
      SUM2 = 0.0
      DO 996 N = 1,NITER
      SUM1 = SUM1+((X1*X1/N)+2.0*X1/(N*N)+2.0/(N*N*N))*EXP(-N*X1)
! C      WRITE(*,*) 'VALUE OF SUM1 AS IT GOES THROUGH ITERATION = ',SUM1 
      SUM2 = SUM2+((X2*X2/N)+2.0*X2/(N*N)+2.0/(N*N*N))*EXP(-N*X2) 
 996  END DO
C NOW CALCULATE DIFFERNCE TO GET BAND RADIANCE
      SUMBB = SUM2 - SUM1
      SUMBB = SUMBB*(2.0*CSI)*(BOLTZ*T/(HPLANCK*CSI))**3
C NOW DIVIDE BY 10000 IN ORDER TO GET PHOTONS CM-2 S-1 STER-1
      SUMBB = SUMBB*1.0E-4
!       WRITE (*,*) 'BAND RADIANCE USING FORMULAE = ', SUMBB, 'Photons cm-
!      *2 s-1 sr-1'

C***
C***  THIS HAS BEEN CHECKED AGAINST THE SPECTRAL CALCULATOR WEBSITE AND IS CORRECT ANSWER

C NOW SCALE TO RADIUS
!       write(*,*) 'radius = ',radius
      SUMBB = PIE*SUMBB/(RADIUS*RADIUS)

      GETGSTAR = SUMBB/SUMDRAINETOT
      WRITE (3,297) T, GETGSTAR
C 777  end do  
 
 296  FORMAT(2X,'TEMP_BB',5X,'BB/DRAINE')    
 297  FORMAT(2X,F6.1,5X,1PE11.4)        
 298  FORMAT(2X,//,'** PHOTO RATES (S-1) AT 50 R_* IN ',A,' **' 
     *  //)
 998  FORMAT (2X,'DRAINE INTERSTELLAR UV FIELD (VAN DISHOECK 1988)')
 999  FORMAT (2X, 'RATIO OF STELLAR BB TO DRAINE FIELD - INTEGRATED
     * 912-2050A'//'BLACKBODY RADIATION FIELD AT TEMPERATURE T_BB
     * AND RADIUS =',F5.1,1X,'STELLAR RADII'//)

      RETURN
      END


      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION TO CALCULATE SELF-SHIELDED CO PHOTODISSOCIATION RATE   C 
C  USING ONE-BAND APPROXIMATION                                    C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETCOR(H2COL,XCO,V,AUV)
C  H2COL = RADIAL H2 COLUMN DENSITY (OUTWARDS)
C  XCO = CO/H2 FRACTIONAL ABUNDANCE
C  V = OUTFLOW VELOCITY
C  AUV = RADIAL EXTINCTION AT 1000 ANGSTROMS

      DOUBLE PRECISION FRACE,LAMDAE,FOSCE,BANDS,GE0,TAUE,GAMMAD,
     *   BETAE,H2COL,XCO,V,AUV,EARG

C  FRACTIONAL POPULATION OF LOWER LEVEL
      FRACE = 1.0/3.0
C  EFFECTIVE WAVELENGTH (IN CM)
      LAMDAE = 1000.0*1.0E-08
C  EFFECTIVE DISSOCIATIVE OSCILLATOR STRENGTH
      FOSCE = 0.017
C  EFFECTIVE NUMBER OF BANDS
      BANDS = 1.0
C  UNSHIELDED PHOTODISSOCIATION RATE OF CO
      GE0 = 2.0E-10
C  CALCULATE EFFECTIVE OPTICAL DEPTH OF CO AT RADIUS
      TAUE = 0.0265*FRACE*FOSCE*LAMDAE*H2COL*XCO*(1.0/V)
C  CALCULATE CONTINUUM SHIELDING BY DUST (MORRIS AND JURA)
      GAMMAD = EARG(-1.644*(AUV**0.86))
C  THE FOLLOWING IS THE MORRIS/JURA APPROXIMATION TO THE FULL INTEGRAL
      BETAE = (1.0-EARG(-1.5*TAUE))/(1.5*TAUE)
C  CALCULATE CO PHOTODISSOCIATION RATE
      GETCOR = GE0*BETAE*GAMMAD*BANDS
c       write(19,*) GETCOR,H2COL,AUV,TAUE,
c      *     BETAE,GAMMAD,XCO
c       write(19,*) BETAE,TAUE,H2COL,XCO,(1.0/V),FRACE,FOSCE,LAMDAE
      RETURN
      END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION FOR THE H2 NUMBER DENSITY AS A FUNCTION OF RADIUS  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETHNR(RADIUS)
      DOUBLE PRECISION RADIUS,PI,MLOSS,MSUN,SEC,MU,MH,V,KB
      COMMON/BL2/ MLOSS,V
      COMMON/BLC/ PI,MH,MU,KB

      MSUN = 1.989E33
      SEC = 31557600.0

      GETHNR=(MLOSS*MSUN/SEC)/(4.0*PI*(RADIUS**2.0)*MU*MH*V)

      RETURN
      END

! CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      
! C FUNCTION FOR THE VELOCITY PROFILE C
! C (SIMPLIFIED BETA LAW)             C     
! CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!       
!       DOUBLE PRECISION FUNCTION GETVEL(RADIUS)
!       DOUBLE PRECISION RADIUS
!       DOUBLE PRECISION R_START,BETA,V_INF,RINIT,RFINAL
!       DOUBLE PRECISION V0
!       CHARACTER*500 VELMODER_STAR
!       
! C BETA LAW FOR VELOCITY
!       IF (VELMODE.EQ.'BETA') THEN
! C 	VELOCITY BEFORE START RADIUS WIND = VELOCITY AT START RADIUS
! C     SHOULD BE ADAPTED TO BETA LAW JOINING THE VELOCITY AT R_START
!       V0 = V_INF/2.0
!       	IF (RADIUS.LT.R_START) THEN
!            	GETVEL = V0
!       	ELSE 
!            	GETVEL = V0+(V_INF-V0)*(1.0-(R_START/RADIUS))**BETA
!       	ENDIF
! C ADOPT VELOCITY PROFILE AGUNDEZ ET AL. 2010      	
!       ELSE IF (VELMODE.EQ.'AGUNDEZ') THEN
!       	IF (RADIUS.LT.R_START) THEN
!       		GETVEL = 5E5
!       	ELSE
!       		GETVEL = 15E5
!       	ENDIF
! 
!       ENDIF
!       
!       
!       
!       RETURN 
!       END

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION FOR THE TEMPERATURE AS A FUNCTION OF RADIUS  C
C  (FIT TO CROSAS & MENTEN 1997), MIN = 10 K             C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETTEM(R)
      DOUBLE PRECISION R,R_STAR,T_STAR,EPSIL,R_EPSIL,S_DUST,T_DUST
      CHARACTER*500 TEMPMODE
      COMMON/BLTEMP/ R_STAR,T_STAR,EPSIL,TEMPMODE,R_EPSIL,S_DUST,
     *   T_DUST
      
C ORIGINAL TEMPERATURE PROFILE      
      IF (TEMPMODE.EQ.'CROSAS') THEN
      GETTEM = 128.0*(1.0E15/R)**4.7 + 447.0*(1.0E15/R)**1.05
      IF (GETTEM.GT.1500.0) THEN
      GETTEM = 1500.0
      END IF
      
      ELSE IF (TEMPMODE.EQ.'EPSILON') THEN
      GETTEM = T_STAR*(R/R_EPSIL)**(-EPSIL)
      
      ELSE IF (TEMPMODE.EQ.'CWLEO') THEN
        IF (R.LT.9.*R_EPSIL) THEN
        GETTEM = T_STAR*(R/R_EPSIL)**(-0.58)
        END IF
        IF ((R.GE.9.*R_EPSIL) .AND. 
     *  (R.LT.65.*R_EPSIL)) THEN
        GETTEM = T_STAR*(9.**-0.58)/(9.**-0.40)*(R/R_EPSIL)**(-0.40)
        END IF
        IF (R.GE.65.*R_EPSIL) THEN
        FRACT = T_STAR*(65.)**(-0.4)/(T_STAR*(65.)**(-1.2))
        GETTEM = T_STAR*(9.**-0.58)/(9.**-0.40)*
     *  (65.**-0.4)/(65.**-1.2)*(R/R_EPSIL)**(-1.2)
        END IF
        
      ELSE IF (TEMPMODE.EQ.'AGUNDEZ') THEN
        IF (R.LT.75.*R_EPSIL) THEN
        GETTEM = T_STAR*(R/R_EPSIL)**(-0.55)
        END IF
        IF ((R.GE.75.*R_EPSIL) .AND. 
     *     (R.LT.200.*R_EPSIL)) THEN
        GETTEM = T_STAR*(75.**-0.55)/(75.**-0.85)*
     *     (R/R_EPSIL)**(-0.85)
        END IF
        IF (R.GE.200.*R_EPSIL) THEN
        FRACT = T_STAR*(65.)**(-0.4)/(T_STAR*(65.)**(-1.2))
        GETTEM = T_STAR*(75.**-0.55)/(75.**-0.85)*
     *    (200.**-0.85)/(200.**-1.4)*(R/R_EPSIL)**(-1.4)
        END IF
      
      ENDIF 
      
      IF (GETTEM.LT.10.0) GETTEM = 10.0
!       IF (GETTEM.GT.1500.0).AND.(TEMPMODE.EQ.'CROSAS') GETTEM = 1500.0

      RETURN
      END

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION FOR THE DUST TEMPERATURE AS A FUNCTION OF RADIUS  C
C  (HABING & OLOFSSON 2003, SECT. 4.1.2., EQ. 55              C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETTEMDUST(R)
      DOUBLE PRECISION R,R_STAR,T_STAR,S_DUST,R_EPSIL,EPSIL,T_DUST
      CHARACTER*500 TEMPMODE
      COMMON/BLTEMP/ R_STAR,T_STAR,EPSIL,TEMPMODE,R_EPSIL,S_DUST,
     *   T_DUST
      
C ORIGINAL TEMPERATURE PROFILE      
      GETTEMDUST = T_DUST*(R_STAR/(2*R))**(2.0/(4+S_DUST))
      
      IF (GETTEMDUST.LT.10.0) GETTEMDUST = 10.0

      RETURN
      END

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION FOR H2 RADIAL COLUMN DENSITY TO OUTER EDGE OF ENVELOPE  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETH2(RADIUS,HNR)
      DOUBLE PRECISION RADIUS,HNR

      GETH2 = HNR*RADIUS

      RETURN
      END

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  RADIAL DISTANCE FROM STAR                                       C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETDIST(RADIUS)
      DOUBLE PRECISION RADIUS
     
      GETDIST = 1.0*RADIUS
      
      RETURN
      END

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  GET MODIFICATION TO THERMAL VELOCITY                            C
C  GAIL& SEDLMAYER 2016, EQ. 6.34, P. 1.9                          C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETMOD(S)
      DOUBLE PRECISION S,PI,MH,MU,KB
     
      COMMON/BLC/ PI,MH,MU,KB

      GETMOD =(0.5*SQRT(PI)*(2.0*S**2.0+1.0)*ERF(S) + S*EXP(-S**2.0))
     *      / (SQRT(PI)*S) 
     
      
      RETURN
      END
      

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  GET SPUTTERING THRESHOLD ENERGY                                 C
C  BOHDANSKY ET AL. 1980, ANDERSEN & BAY 1981                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETETH(MT,MP,EB)
      DOUBLE PRECISION PI,MH,MU,KB,MT,MP,RATIO,GAMMA,EB,EB_EV
      
C   MT = MOLECULAR MASS TARGET SPECIES
C   MP = MOLECULAR MASS PROJECTILE (H2, HE)
C   EB = BINDING ENERGY IN K
C   GETETH = THRESHOLD ENERGY FOR SPUTTERING IN EV

      COMMON/BLC/ PI,MH,MU,KB
      
C   CONVERT BINDING ENERGY TO EV      
      EB_EV = EB*8.621738E-5 
      
      RATIO = MP/MT
      GAMMA = 4.*MP*MT/((MP+MT)**2.)
      
      IF (RATIO.LE.0.3) THEN
        GETETH = EB_EV/(GAMMA*(1.-GAMMA))
      ELSE
        GETETH = 8.*EB_EV*(MP/MT)**(1./3.)
      END IF
      
      RETURN
      END
      

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  GET ATOMIC NUMBER OF BALLISTIC PARTICLE                        C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      DOUBLE PRECISION FUNCTION GETZP(BALLISTIC)
      CHARACTER*500 BALLISTIC
      
!       WRITE(*,*) BALLISTIC
!       IF (BALLISTIC.EQ.'H2') THEN
!       WRITE(*,*) 'OK'
!         GETZP = 1.
!       END IF
      
      IF (BALLISTIC.EQ.'H2') THEN
        GETZP = 1.
      ELSE IF (BALLISTIC.EQ.'He') THEN
        GETZP = 2.
      ELSE IF (BALLISTIC.EQ.'CO') THEN
        GETZP = 7.
      ELSE IF (BALLISTIC.EQ.'N2') THEN
        GETZP = 7.
      ELSE IF (BALLISTIC.EQ.'H2O') THEN
        GETZP = 3.33
      ELSE IF (BALLISTIC.EQ.'C2H2') THEN
        GETZP = 3.5
      ELSE IF (BALLISTIC.EQ.'HCN') THEN
        GETZP = 4.67
      END IF
      
      RETURN
      END

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  GET ATOMIC MASS OF BALLISTIC PARTICLE                           C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      DOUBLE PRECISION FUNCTION GETMP(BALLISTIC)
      CHARACTER*500 BALLISTIC
      INTEGER IDGCHEM,ILH,IER,IGRAINSIZEDIST,ISCALEAUV,
     *    ISPUTTERINGWOITKE,ISPUTTERINGTIELENS,SPUTTIELENS,SPUTCO,
     *    SPUTN2,SPUTH2O,SPUTHCN,SPUTC2H2,SPUTFACTORTWO,RATEVR,
     *    IGRAINPHOTON
      DOUBLE PRECISION BRANCH
      COMMON/SWITCH/ IDGCHEM,ILH,IER,IGRAINSIZEDIST,ISCALEAUV,
     *    ISPUTTERINGWOITKE,ISPUTTERINGTIELENS,SPUTTIELENS,SPUTCO,
     *    SPUTN2,SPUTFACTORTWO,RATEVR,BRANCH,IGRAINPHOTON

      
      IF (ISPUTTERINGTIELENS.EQ.1) THEN 
!       "MEAN" MOLECULAR MASS FOR SPUTTERING TIELENS
        IF (BALLISTIC.EQ.'H2') THEN
          GETMP = 2./2.
        ELSE IF (BALLISTIC.EQ.'He') THEN
          GETMP = 4.
        ELSE IF (BALLISTIC.EQ.'CO') THEN
          GETMP = 28./2.
        ELSE IF (BALLISTIC.EQ.'N2') THEN
          GETMP = 28./2.
        ELSE IF (BALLISTIC.EQ.'H2O') THEN
          GETMP = 18./3.
        ELSE IF (BALLISTIC.EQ.'C2H2') THEN
          GETMP = 26./4.
        ELSE IF (BALLISTIC.EQ.'HCN') THEN
          GETMP = 27./3.
        END IF
      
      ELSE IF (ISPUTTERINGWOITKE.EQ.1) THEN
!       "NORMAL" MOLECULAR MASS FOR SPUTTERING WOITKE
        IF (BALLISTIC.EQ.'H2') THEN
          GETMP = 2.
        ELSE IF (BALLISTIC.EQ.'He') THEN
          GETMP = 4.
        ELSE IF (BALLISTIC.EQ.'CO') THEN
          GETMP = 28.
        ELSE IF (BALLISTIC.EQ.'N2') THEN
          GETMP = 28.
        ELSE IF (BALLISTIC.EQ.'H2O') THEN
          GETMP = 18.
        ELSE IF (BALLISTIC.EQ.'C2H2') THEN
          GETMP = 26.
        ELSE IF (BALLISTIC.EQ.'HCN') THEN
          GETMP = 27.
        END IF
        
      END IF
      
      
      RETURN
      END
            
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION FOR EXTINCTION AT 1000 ANGSTROMS (NEJAD AND MILLAR)   C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETAUV(H2COL,RADIUS)
C  H2COL = RADIAL H2 COLUMN DENSITY (OUTWARDS)
      CHARACTER*500 CLUMPMODE,TEMPMODE
      DOUBLE PRECISION H2COL,AUV_AV,GAMCO,ALBEDO,ZETA,A_G,X_G,Y(1000),
     *    GSTAR,RSCALE,RDUST,DELTA_AUV,AUV,K,H,A,B,C,RADIUS,TEFF,
     *    TEFF_R,TEFF_INF,R_STAR,PI,MH,MU,KB,T_STAR,EPSIL,
     *    FVOL,L,FIC,BFILL,CFILL,R,R_EPSIL,S_DUST,T_DUST,
     *    DTG,GRAINEXP,AMAX,AMIN,RHODUST,VDRIFT,RATEVR,BRANCH,AGMEAN,
     *    GBIN,RBIN,TBIN
      INTEGER ICO,ICLUMP,IDGCHEM,ILH,IER,IGRAINSIZEDIST,
     *    ISCALEAUV,ISPUTTERINGWOITKE,ISPUTTERINGTIELENS,SPUTCO,SPUTN2,
     *    SPUTFACTORTWO,IGRAIN0,IGRAINPHOTON,ISTELLAR,IBIN
      COMMON/BL3/ Y,X_G,A_G,ZETA,ALBEDO,GAMCO,AUV_AV,ICO,GSTAR,RSCALE,
     *   RDUST,DELTA_AUV,DTG,GRAINEXP,AMAX,AMIN,RHODUST,VDRIFT,AGMEAN,
     *   GBIN,ISTELLAR,IBIN,RBIN,TBIN
      COMMON/BLC/ PI,MH,MU,KB
      COMMON/CLM/ CLUMPMODE,ICLUMP,FVOL,L,FIC
      COMMON/BLTEMP/ R_STAR,T_STAR,EPSIL,TEMPMODE,R_EPSIL,S_DUST,
     *   T_DUST
      COMMON/SWITCH/ IDGCHEM,ILH,IER,IGRAINSIZEDIST,ISCALEAUV,
     *    ISPUTTERINGWOITKmodel E,ISPUTTERINGTIELENS,SPUTTIELENS,SPUTCO,
     *    SPUTN2,SPUTFACTORTWO,RATEVR,BRANCH,IGRAINPHOTON
C	Column density H2/A(V) = 1.87E21 atoms cm-2 mag-1
C	http://adsabs.harvard.edu/abs/1995A&A...293..889P
      
c       AUV = (1/DTG)*1e-2*(AUV_AV*2.0*H2COL)/1.87E21
c       removed factor 2!
      
      
      IF (IGRAINSIZEDIST.EQ.1.AND.ISCALEAUV.EQ.1) THEN
      AUV = (1/DTG)*1e-2*(AUV_AV*H2COL)/1.87E21
      ELSE IF (IGRAINSIZEDIST.EQ.0.AND.ISCALEAUV.EQ.1) THEN
!       DTG = 4 PI X_G RHODUST A_G**3/(3*MASS H2)
      AUV = (1/DTG)*1e-2*(AUV_AV*H2COL)/1.87E21
      ELSE
      AUV = (AUV_AV*H2COL)/1.87E21
      END IF
      
      IF (CLUMPMODE.EQ.'POROSITY') THEN            

      R = RADIUS      
      H = L*(R_STAR**(-2.0/3.0))/FVOL 
      BFILL = 1.0-(1.0-FVOL)*FIC  
C DEFINE FOUR TERMS FOR USE IN CALCULATING TAU_EFF/AUV_EFF
      CFILL = AUV*H*BFILL
      F1 = ATAN(1.0 + (SQRT(2.0)*(R**(1.0/12.0))*((CFILL)**(-0.25))))
      F2 = ATAN(1.0 - (SQRT(2.0)*(R**(1.0/12.0))*((CFILL)**(-0.25))))
      F3 = 0.5*LOG(SQRT(CFILL*R)+SQRT(2.0)*((CFILL)**0.25)*
     *  (R**(7.0/12.0)) + R**(2.0/3.0))
      F4 = 0.5*LOG(SQRT(CFILL*R)-SQRT(2.0)*((CFILL)**0.25)*
     *  (R**(7.0/12.0)) + R**(2.0/3.0))
      F5 = (3.0/(2.0*SQRT(2.0)))*((AUV*R)**(0.25))*
     *   ((H*BFILL)**(-0.75))*(1.0-FIC)
C  INTERCLUMP
      GETAUV = AUV*FIC + F5*(PI-F1+F2-F3+F4)

      
      ELSE
         GETAUV = AUV
      END IF
      
      
      
      RETURN
      END


      
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION TO CALCULATE RADIATION FIELD STRENGTH AT 1000 A  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETRAD(AUV)
C  AUV = RADIAL EXTINCTION AT 1000 ANGSTROMS

      DOUBLE PRECISION AUV,CETA(6),GA(6),W(6),PI,SUM,EARG
      INTEGER I

      PI = 4.*ATAN(1.)

C   SET COEFFICIENTS FOR CALCULATION OF RADIATION FIELD
      W(1) = 0.17132449
      W(2) = 0.36076157
      W(3) = 0.46791393
      W(4) = W(1)
      W(5) = W(2)
      W(6) = W(3)
      GA(1) = 0.93246951
      GA(2) = 0.66120939
      GA(3) = 0.23861919
      GA(4) = -GA(1)
      GA(5) = -GA(2)
      GA(6) = -GA(3)

      SUM = 0.0
      DO I=1,6
      CETA(I) = (PI*GA(I)+PI)/2.0
      SUM=SUM+(W(I)*(SIN(CETA(I))*EXP((-AUV*CETA(I))/SIN(CETA(I)))))
      
      END DO
      GETRAD = (PI/4.0)*SUM
      
      
      RETURN
      END
     
     
     
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION TO CALCULATE DUST NUMBER DENSITY  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      DOUBLE PRECISION FUNCTION GETDUSTDENS(HNR)
C  AUV = RADIAL EXTINCTION AT 1000 ANGSTROMS

      DOUBLE PRECISION Y(1000),X_G,A_G,ZETA,ALBEDO,GAMCO,AUV_AV,ICO,
     *   GSTAR,RSCALE,RDUST,DELTA_AUV,DTG,GRAINEXP,AMAX,AMIN,RHODUST,
     *   VDRIFT,PI,MH,MU,KB,HNR,AGMEAN,GBIN,RBIN,TBIN
      INTEGER ISTELLAR,IBIN
      
      COMMON/BL3/ Y,X_G,A_G,ZETA,ALBEDO,GAMCO,AUV_AV,ICO,GSTAR,RSCALE,
     *   RDUST,DELTA_AUV,DTG,GRAINEXP,AMAX,AMIN,RHODUST,VDRIFT,AGMEAN,
     *   GBIN,ISTELLAR,IBIN,RBIN,TBIN
      COMMON/BLC/ PI,MH,MU,KB

      CTEGRAIN = 3.*(4.+GRAINEXP)*DTG*MU*MH*HNR /
     *    (4.*PI*RHODUST*(AMAX**(4.+GRAINEXP)-AMIN**(4.+GRAINEXP)))
      
      write(*,*) CTEGRAIN
      
      GETDUSTDENS = CTEGRAIN/(1.+GRAINEXP) *
     *     (AMAX**(1.+GRAINEXP)-AMIN**(1.+GRAINEXP))
     
      RETURN
      END
  

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  ALTERNATIVE EXP FUNCTION TO AVOID EXPONENTIAL ERROR  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION EARG(TZZ)
      DOUBLE PRECISION TZZ

      IF(TZZ.LE.-20.0) EARG = 0.0
      IF(TZZ.GT.-20.0.AND.TZZ.LT.710.) EARG = EXP(TZZ)
      IF(TZZ.GE.710.) EARG = 8D307
      RETURN
      END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C   DUMMY ROUTINE TO SATISFY LOADER   C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE PEDERV(N,T,Y,PD,N0)

      RETURN
      END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C   ERROR HANDLING ROUTINE  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE DIE(ERROR)
      CHARACTER*59 ERROR

      WRITE(*,*) "ERROR: ",ERROR
      WRITE(*,*) "...Aborting......................."

      STOP
      END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  BEGIN NUMERICAL SUBROUTINES REQUIRED BY DVODE  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      subroutine  dcopy(n,dx,incx,dy,incy)
c
c     copies a vector, x, to a vector, y.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double precision dx(*),dy(*)
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,7)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dx(i)
   30 continue
      if( n .lt. 7 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,7
        dy(i) = dx(i)
        dy(i + 1) = dx(i + 1)
        dy(i + 2) = dx(i + 2)
        dy(i + 3) = dx(i + 3)
        dy(i + 4) = dx(i + 4)
        dy(i + 5) = dx(i + 5)
        dy(i + 6) = dx(i + 6)
   50 continue
      return
      end
C
C---
C
      subroutine daxpy(n,da,dx,incx,dy,incy)
c
c     constant times a vector plus a vector.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double precision dx(*),dy(*),da
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if (da .eq. 0.0d0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dy(iy) + da*dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,4)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dy(i) + da*dx(i)
   30 continue
      if( n .lt. 4 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,4
        dy(i) = dy(i) + da*dx(i)
        dy(i + 1) = dy(i + 1) + da*dx(i + 1)
        dy(i + 2) = dy(i + 2) + da*dx(i + 2)
        dy(i + 3) = dy(i + 3) + da*dx(i + 3)
   50 continue
      return
      end
C
C---
C
      subroutine  dscal(n,da,dx,incx)
c
c     scales a vector by a constant.
c     uses unrolled loops for increment equal to one.
c     jack dongarra, linpack, 3/11/78.
c     modified 3/93 to return if incx .le. 0.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double precision da,dx(*)
      integer i,incx,m,mp1,n,nincx
c
      if( n.le.0 .or. incx.le.0 )return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      nincx = n*incx
      do 10 i = 1,nincx,incx
        dx(i) = da*dx(i)
   10 continue
      return
c
c        code for increment equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,5)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dx(i) = da*dx(i)
   30 continue
      if( n .lt. 5 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,5
        dx(i) = da*dx(i)
        dx(i + 1) = da*dx(i + 1)
        dx(i + 2) = da*dx(i + 2)
        dx(i + 3) = da*dx(i + 3)
        dx(i + 4) = da*dx(i + 4)
   50 continue
      return
      end
C
C---
C
      subroutine dgefa(a,lda,n,ipvt,info)
      integer lda,n,ipvt(n),info
      double precision a(lda,n)
c
c     dgefa factors a double precision matrix by gaussian elimination.
c
c     dgefa is usually called by dgeco, but it can be called
c     directly with a saving in time if  rcond  is not needed.
c     (time for dgeco) = (1 + 9/n)*(time for dgefa) .
c
c     on entry
c
c        a       double precision(lda, n)
c                the matrix to be factored.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c     on return
c
c        a       an upper triangular matrix and the multipliers
c                which were used to obtain it.
c                the factorization can be written  a = l*u  where
c                l  is a product of permutation and unit lower
c                triangular matrices and  u  is upper triangular.
c
c        ipvt    integer(n)
c                an integer vector of pivot indices.
c
c        info    integer
c                = 0  normal value.
c                = k  if  u(k,k) .eq. 0.0 .  this is not an error
c                     condition for this subroutine, but it does
c                     indicate that dgesl or dgedi will divide by zero
c                     if called.  use  rcond  in dgeco for a reliable
c                     indication of singularity.
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas daxpy,dscal,idamax
c
c     internal variables
c
      double precision t
      integer idamax,j,k,kp1,l,nm1
c
c
c     gaussian elimination with partial pivoting
c
      info = 0
      nm1 = n - 1
      if (nm1 .lt. 1) go to 70
      do 60 k = 1, nm1
         kp1 = k + 1
c
c        find l = pivot index
c
         l = idamax(n-k+1,a(k,k),1) + k - 1
         ipvt(k) = l
c
c        zero pivot implies this column already triangularized
c
         if (a(l,k) .eq. 0.0d0) go to 40
c
c           interchange if necessary
c
            if (l .eq. k) go to 10
               t = a(l,k)
               a(l,k) = a(k,k)
               a(k,k) = t
   10       continue
c
c           compute multipliers
c
            t = -1.0d0/a(k,k)
            call dscal(n-k,t,a(k+1,k),1)
c
c           row elimination with column indexing
c
            do 30 j = kp1, n
               t = a(l,j)
               if (l .eq. k) go to 20
                  a(l,j) = a(k,j)
                  a(k,j) = t
   20          continue
               call daxpy(n-k,t,a(k+1,k),1,a(k+1,j),1)
   30       continue
         go to 50
   40    continue
            info = k
   50    continue
   60 continue
   70 continue
      ipvt(n) = n
      if (a(n,n) .eq. 0.0d0) info = n
      return
      end
C
C---
C
      subroutine dgbsl(abd,lda,n,ml,mu,ipvt,b,job)
      integer lda,n,ml,mu,ipvt(1),job
      double precision abd(lda,1),b(1)
c
c     dgbsl solves the double precision band system
c     a * x = b  or  trans(a) * x = b
c     using the factors computed by dgbco or dgbfa.
c
c     on entry
c
c        abd     double precision(lda, n)
c                the output from dgbco or dgbfa.
c
c        lda     integer
c                the leading dimension of the array  abd .
c
c        n       integer
c                the order of the original matrix.
c
c        ml      integer
c                number of diagonals below the main diagonal.
c
c        mu      integer
c                number of diagonals above the main diagonal.
c
c        ipvt    integer(n)
c                the pivot vector from dgbco or dgbfa.
c
c        b       double precision(n)
c                the right hand side vector.
c
c        job     integer
c                = 0         to solve  a*x = b ,
c                = nonzero   to solve  trans(a)*x = b , where
c                            trans(a)  is the transpose.
c
c     on return
c
c        b       the solution vector  x .
c
c     error condition
c
c        a division by zero will occur if the input factor contains a
c        zero on the diagonal.  technically this indicates singularity
c        but it is often caused by improper arguments or improper
c        setting of lda .  it will not occur if the subroutines are
c        called correctly and if dgbco has set rcond .gt. 0.0
c        or dgbfa has set info .eq. 0 .
c
c     to compute  inverse(a) * c  where  c  is a matrix
c     with  p  columns
c           call dgbco(abd,lda,n,ml,mu,ipvt,rcond,z)
c           if (rcond is too small) go to ...
c           do 10 j = 1, p
c              call dgbsl(abd,lda,n,ml,mu,ipvt,c(1,j),0)
c        10 continue
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas daxpy,ddot
c     fortran min0
c
c     internal variables
c
      double precision ddot,t
      integer k,kb,l,la,lb,lm,m,nm1
c
      m = mu + ml + 1
      nm1 = n - 1
      if (job .ne. 0) go to 50
c
c        job = 0 , solve  a * x = b
c        first solve l*y = b
c
         if (ml .eq. 0) go to 30
         if (nm1 .lt. 1) go to 30
            do 20 k = 1, nm1
               lm = min0(ml,n-k)
               l = ipvt(k)
               t = b(l)
               if (l .eq. k) go to 10
                  b(l) = b(k)
                  b(k) = t
   10          continue
               call daxpy(lm,t,abd(m+1,k),1,b(k+1),1)
   20       continue
   30    continue
c
c        now solve  u*x = y
c
         do 40 kb = 1, n
            k = n + 1 - kb
            b(k) = b(k)/abd(m,k)
            lm = min0(k,m) - 1
            la = m - lm
            lb = k - lm
            t = -b(k)
            call daxpy(lm,t,abd(la,k),1,b(lb),1)
   40    continue
      go to 100
   50 continue
c
c        job = nonzero, solve  trans(a) * x = b
c        first solve  trans(u)*y = b
c
         do 60 k = 1, n
            lm = min0(k,m) - 1
            la = m - lm
            lb = k - lm
            t = ddot(lm,abd(la,k),1,b(lb),1)
            b(k) = (b(k) - t)/abd(m,k)
   60    continue
c
c        now solve trans(l)*x = y
c
         if (ml .eq. 0) go to 90
         if (nm1 .lt. 1) go to 90
            do 80 kb = 1, nm1
               k = n - kb
               lm = min0(ml,n-k)
               b(k) = b(k) + ddot(lm,abd(m+1,k),1,b(k+1),1)
               l = ipvt(k)
               if (l .eq. k) go to 70
                  t = b(l)
                  b(l) = b(k)
                  b(k) = t
   70          continue
   80       continue
   90    continue
  100 continue
      return
      end
C
C---
C
      subroutine dgesl(a,lda,n,ipvt,b,job)
      integer lda,n,ipvt(n),job
      double precision a(lda,n),b(n)
c
c     dgesl solves the double precision system
c     a * x = b  or  trans(a) * x = b
c     using the factors computed by dgeco or dgefa.
c
c     on entry
c
c        a       double precision(lda, n)
c                the output from dgeco or dgefa.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c        ipvt    integer(n)
c                the pivot vector from dgeco or dgefa.
c
c        b       double precision(n)
c                the right hand side vector.
c
c        job     integer
c                = 0         to solve  a*x = b ,
c                = nonzero   to solve  trans(a)*x = b  where
c                            trans(a)  is the transpose.
c
c     on return
c
c        b       the solution vector  x .
c
c     error condition
c
c        a division by zero will occur if the input factor contains a
c        zero on the diagonal.  technically this indicates singularity
c        but it is often caused by improper arguments or improper
c        setting of lda .  it will not occur if the subroutines are
c        called correctly and if dgeco has set rcond .gt. 0.0
c        or dgefa has set info .eq. 0 .
c
c     to compute  inverse(a) * c  where  c  is a matrix
c     with  p  columns
c           call dgeco(a,lda,n,ipvt,rcond,z)
c           if (rcond is too small) go to ...
c           do 10 j = 1, p
c              call dgesl(a,lda,n,ipvt,c(1,j),0)
c        10 continue
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas daxpy,ddot
c
c     internal variables
c
      double precision ddot,t
      integer k,kb,l,nm1
c
      nm1 = n - 1
      if (job .ne. 0) go to 50
c
c        job = 0 , solve  a * x = b
c        first solve  l*y = b
c
         if (nm1 .lt. 1) go to 30
         do 20 k = 1, nm1
            l = ipvt(k)
            t = b(l)
            if (l .eq. k) go to 10
               b(l) = b(k)
               b(k) = t
   10       continue
            call daxpy(n-k,t,a(k+1,k),1,b(k+1),1)
   20    continue
   30    continue
c
c        now solve  u*x = y
c
         do 40 kb = 1, n
            k = n + 1 - kb
            b(k) = b(k)/a(k,k)
            t = -b(k)
            call daxpy(k-1,t,a(1,k),1,b(1),1)
   40    continue
      go to 100
   50 continue
c
c        job = nonzero, solve  trans(a) * x = b
c        first solve  trans(u)*y = b
c
         do 60 k = 1, n
            t = ddot(k-1,a(1,k),1,b(1),1)
            b(k) = (b(k) - t)/a(k,k)
   60    continue
c
c        now solve trans(l)*x = y
c
         if (nm1 .lt. 1) go to 90
         do 80 kb = 1, nm1
            k = n - kb
            b(k) = b(k) + ddot(n-k,a(k+1,k),1,b(k+1),1)
            l = ipvt(k)
            if (l .eq. k) go to 70
               t = b(l)
               b(l) = b(k)
               b(k) = t
   70       continue
   80    continue
   90    continue
  100 continue
      return
      end
C
C---
C
      subroutine dgbfa(abd,lda,n,ml,mu,ipvt,info)
      integer lda,n,ml,mu,ipvt(1),info
      double precision abd(lda,1)
c
c     dgbfa factors a double precision band matrix by elimination.
c
c     dgbfa is usually called by dgbco, but it can be called
c     directly with a saving in time if  rcond  is not needed.
c
c     on entry
c
c        abd     double precision(lda, n)
c                contains the matrix in band storage.  the columns
c                of the matrix are stored in the columns of  abd  and
c                the diagonals of the matrix are stored in rows
c                ml+1 through 2*ml+mu+1 of  abd .
c                see the comments below for details.
c
c        lda     integer
c                the leading dimension of the array  abd .
c                lda must be .ge. 2*ml + mu + 1 .
c
c        n       integer
c                the order of the original matrix.
c
c        ml      integer
c                number of diagonals below the main diagonal.
c                0 .le. ml .lt. n .
c
c        mu      integer
c                number of diagonals above the main diagonal.
c                0 .le. mu .lt. n .
c                more efficient if  ml .le. mu .
c     on return
c
c        abd     an upper triangular matrix in band storage and
c                the multipliers which were used to obtain it.
c                the factorization can be written  a = l*u  where
c                l  is a product of permutation and unit lower
c                triangular matrices and  u  is upper triangular.
c
c        ipvt    integer(n)
c                an integer vector of pivot indices.
c
c        info    integer
c                = 0  normal value.
c                = k  if  u(k,k) .eq. 0.0 .  this is not an error
c                     condition for this subroutine, but it does
c                     indicate that dgbsl will divide by zero if
c                     called.  use  rcond  in dgbco for a reliable
c                     indication of singularity.
c
c     band storage
c
c           if  a  is a band matrix, the following program segment
c           will set up the input.
c
c                   ml = (band width below the diagonal)
c                   mu = (band width above the diagonal)
c                   m = ml + mu + 1
c                   do 20 j = 1, n
c                      i1 = max0(1, j-mu)
c                      i2 = min0(n, j+ml)
c                      do 10 i = i1, i2
c                         k = i - j + m
c                         abd(k,j) = a(i,j)
c                10    continue
c                20 continue
c
c           this uses rows  ml+1  through  2*ml+mu+1  of  abd .
c           in addition, the first  ml  rows in  abd  are used for
c           elements generated during the triangularization.
c           the total number of rows needed in  abd  is  2*ml+mu+1 .
c           the  ml+mu by ml+mu  upper left triangle and the
c           ml by ml  lower right triangle are not referenced.
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas daxpy,dscal,idamax
c     fortran max0,min0
c
c     internal variables
c
      double precision t
      integer i,idamax,i0,j,ju,jz,j0,j1,k,kp1,l,lm,m,mm,nm1
c
c
      m = ml + mu + 1
      info = 0
c
c     zero initial fill-in columns
c
      j0 = mu + 2
      j1 = min0(n,m) - 1
      if (j1 .lt. j0) go to 30
      do 20 jz = j0, j1
         i0 = m + 1 - jz
         do 10 i = i0, ml
            abd(i,jz) = 0.0d0
   10    continue
   20 continue
   30 continue
      jz = j1
      ju = 0
c
c     gaussian elimination with partial pivoting
c
      nm1 = n - 1
      if (nm1 .lt. 1) go to 130
      do 120 k = 1, nm1
         kp1 = k + 1
c
c        zero next fill-in column
c
         jz = jz + 1
         if (jz .gt. n) go to 50
         if (ml .lt. 1) go to 50
            do 40 i = 1, ml
               abd(i,jz) = 0.0d0
   40       continue
   50    continue
c
c        find l = pivot index
c
         lm = min0(ml,n-k)
         l = idamax(lm+1,abd(m,k),1) + m - 1
         ipvt(k) = l + k - m
c
c        zero pivot implies this column already triangularized
c
         if (abd(l,k) .eq. 0.0d0) go to 100
c
c           interchange if necessary
c
            if (l .eq. m) go to 60
               t = abd(l,k)
               abd(l,k) = abd(m,k)
               abd(m,k) = t
   60       continue
c
c           compute multipliers
c
            t = -1.0d0/abd(m,k)
            call dscal(lm,t,abd(m+1,k),1)
c
c           row elimination with column indexing
c
            ju = min0(max0(ju,mu+ipvt(k)),n)
            mm = m
            if (ju .lt. kp1) go to 90
            do 80 j = kp1, ju
               l = l - 1
               mm = mm - 1
               t = abd(l,j)
               if (l .eq. mm) go to 70
                  abd(l,j) = abd(mm,j)
                  abd(mm,j) = t
   70          continue
               call daxpy(lm,t,abd(m+1,k),1,abd(mm+1,j),1)
   80       continue
   90       continue
         go to 110
  100    continue
            info = k
  110    continue
  120 continue
  130 continue
      ipvt(n) = n
      if (abd(m,n) .eq. 0.0d0) info = n
      return
      end
C
C---
C
      integer function idamax(n,dx,incx)
c
c     finds the index of element having max. absolute value.
c     jack dongarra, linpack, 3/11/78.
c     modified 3/93 to return if incx .le. 0.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double precision dx(*),dmax
      integer i,incx,ix,n
c
      idamax = 0
      if( n.lt.1 .or. incx.le.0 ) return
      idamax = 1
      if(n.eq.1)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      dmax = dabs(dx(1))
      ix = ix + incx
      do 10 i = 2,n
         if(dabs(dx(ix)).le.dmax) go to 5
         idamax = i
         dmax = dabs(dx(ix))
    5    ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 dmax = dabs(dx(1))
      do 30 i = 2,n
         if(dabs(dx(i)).le.dmax) go to 30
         idamax = i
         dmax = dabs(dx(i))
   30 continue
      return
      end
C
C---
C
      double precision function ddot(n,dx,incx,dy,incy)
c
c     forms the dot product of two vectors.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double precision dx(*),dy(*),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
c
      ddot = 0.0d0
      dtemp = 0.0d0
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dtemp + dx(ix)*dy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      ddot = dtemp
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,5)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dtemp + dx(i)*dy(i)
   30 continue
      if( n .lt. 5 ) go to 60
   40 mp1 = m + 1
      do 50 i = mp1,n,5
        dtemp = dtemp + dx(i)*dy(i) + dx(i + 1)*dy(i + 1) +
     *   dx(i + 2)*dy(i + 2) + dx(i + 3)*dy(i + 3) + dx(i + 4)*dy(i + 4)
   50 continue
   60 ddot = dtemp
      return
      end


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION TO READ RATE FILE AND STORE REQUIRED DATA  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE READR(FRATES,URATES)
C  FRATES = RATE FILE NAME
C  URATES = RATE FILE UNIT NUMBER

      CHARACTER*500 FRATES,SDATA,TOKEN(100),
     *   REACT1(15000),REACT2(15000)
      CHARACTER*12 RE1,RE2,PR1,PR2,PR3,PR4,STAR,RT
      CHARACTER*1 SEPC,SEP
      DOUBLE PRECISION ALF(15000,5),BET(15000,5),GAM(15000,5),
     *   TLOWER(15000,5),TUPPER(15000,5)
      INTEGER RTYPE(15000),NTR(15000),NREAC,I,J,K,L,URATES,ICOR,
     *   ISTRIN,ICOIP

C  COMMON BLOCK FOR REACTION DATA STORAGE
      COMMON/BL4/ ALF,BET,GAM,TLOWER,TUPPER,RTYPE,NTR,NREAC,ICOR,ICOIP,
     *   ICOAP
      COMMON/SPEC/ REACT1,REACT2

C  TOKEN (DATA SUBSTRING SEPARATOR)
      SEPC = ':'
      SEP = SEPC

C  OPEN RATE FILE
      OPEN(UNIT=URATES, FILE=FRATES, STATUS='OLD')

C  OPEN A NEW TEXT FILE TO WRITE THE BASIC REACTION DATA INTO
      OPEN(UNIT=10, FILE='rates.dat')

C  DATA FILE READ LOOP
      J = 0
 100  J = J+1
      READ(URATES,101,END=300) SDATA
 101  FORMAT(A)

C  SEARCH THROUGH DATA STRING (SDATA) AND FIND N SUB-STRINGS (TOKEN(I))

C  INITIALISE TOKEN COUNTER
      I = 0

C  BEGIN TOKENISATION OF THE DATA STRING 
 200  I = I + 1
C  TOKEN IS A STRING (DELIMITED BY ""), WHICH MAY CONTAIN THE SEPARATOR
         IF (SDATA(1:1).EQ.'"') THEN
            SDATA = SDATA(2:)
            SEP = '"'
            ISTRIN = 1
         END IF
C  FIND TOKEN POSITION
         L = INDEX(SDATA,SEP) - 1
C  POSITIVE LENGTH TOKEN
         IF (L.GT.0) THEN
            TOKEN(I) = SDATA(:L)
            L = L + 1
C  ZERO-LENGTH TOKEN
         ELSE IF (L.EQ.0) THEN
            TOKEN(I) = ""
            L = 1
C  NO MORE TOKENS FOUND         
         ELSE IF (L.EQ.-1) THEN
            TOKEN(I) = SDATA(1:)
            L = 0
         END IF
C  REMOVE PREVIOUS TOKEN FROM DATA STRING
         SDATA = SDATA(L+1:)
C  IF TOKEN WAS A STRING THEN REMOVE THE FOLLOWING CHARACTER AS WELL
         IF (ISTRIN.EQ.1) THEN
            ISTRIN = 0
            SEP = SEPC
            SDATA = SDATA(2:)
         END IF        
C  CONTINUE LOOPING UNTIL NO MORE TOKENS REMAIN
      IF (I.LE.100.AND.L.GT.0) GO TO 200 

C  IF LESS THAN 14 TOKENS WERE READ, REACTION DATA IS INCOMPLETE
      IF(I.LT.14) CALL DIE
     *   ("Unable to interpret reaction string -- check rate file     ")


C  STORE INDEX, REACTANTS AND PRODUCTS
      READ(TOKEN(1),*) I
      RT = TOKEN(2)  
      RE1 = TOKEN(3)
      RE2 = TOKEN(4)
      PR1 = TOKEN(5)
      PR2 = TOKEN(6)
      PR3 = TOKEN(7)
      PR4 = TOKEN(8)
      STAR = TOKEN(18)
      
c       write(*,*) RT,RE1,RE2,PR1,PR2,PR3,PR4
C  NUMBER OF TEMPERATURE RANGES
      READ(TOKEN(9),*) NTR(J)
      
C  LOOP THROUGH AND STORE REACTION DATA FOR DIFFERENT TEMPERATURE RANGES
      DO K = 1,NTR(J)
         L = K - 1
         READ(TOKEN(10+9*L),*) ALF(J,K)
         READ(TOKEN(11+9*L),*) BET(J,K)
         READ(TOKEN(12+9*L),*) GAM(J,K)
         READ(TOKEN(13+9*L),*) TLOWER(J,K)
         READ(TOKEN(14+9*L),*) TUPPER(J,K)
      END DO

C  READ IN REACTANTS      
      REACT1(J) = TRIM(RE1)
      REACT2(J) = TRIM(RE2)
      
C  DETERMINE REACTION TYPE:

C  DEFAULT BINARY REACTION
      RTYPE(J) = 0
C  COSMIC RAY REACTION
      IF(RE2.EQ.'CRP') RTYPE(J) = 1
C  COSMIC RAY-INDUCED PHOTON REACTION
      IF(RE2.EQ.'CRPHOT') RTYPE(J) = 2
C  PHOTON REACTION
      IF(RE2.EQ.'PHOTON') RTYPE(J) = 3

C  INTERNAL PHOTONS - CROSS-SECTIONS NOT AVAILABLE
      IF(RT.EQ.'IP'.AND.RE2.EQ.'INPHOTON') RTYPE(J) = 4
C  INTERNAL PHOTONS - CROSS-SECTIONS AVAILABLE
      IF(RT.EQ.'IP'.AND.RE2.EQ.'INPHOTON'.AND.
     *    STAR.EQ.'CWLeo') RTYPE(J) = 5
C  INTERNAL PHOTONS - CROSS-SECTIONS AVAILABLE - IONISATION RATES
      IF(RT.EQ.'IP'.AND.RE2.EQ.'INPHOTON'.AND.
     *     STAR.EQ.'Millar 2018') RTYPE(J) = 5
      
C  COMPANION (DISK) PHOTONS - CROSS-SECTIONS NOT AVAILABLE
      IF(RT.EQ.'AP'.AND.RE2.EQ.'ACPHOTON') RTYPE(J) = 20
C  COMPANION (DISK) PHOTONS - CROSS-SECTIONS AVAILABLE
      IF(RT.EQ.'AP'.AND.RE2.EQ.'ACPHOTON'.AND.
     *     STAR.EQ.'CWLeo') RTYPE(J) = 21
C  COMPANION (DISK) PHOTONS - CROSS-SECTIONS NOT AVAILABLE - IONISATION RATES
      IF(RT.EQ.'AP'.AND.RE2.EQ.'ACPHOTON'.AND.
     *     STAR.EQ.'Millar 2018') RTYPE(J) = 22
      
C  DUST-GRAIN INTERACTION: FREEZE-OUT
      IF(RT.EQ.'FO') RTYPE(J) = 6
C  DUST-GRAIN INTERACTION: THERMAL DESOPRTION
      IF(RT.EQ.'TD') RTYPE(J) = 7
C  DUST-GRAIN INTERACTION: PHOTODESORPTION
      IF(RT.EQ.'PD') RTYPE(J) = 8
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: LANGUIR-HINSHELWOOD
      IF(RT.EQ.'LH') RTYPE(J) = 9
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: ELEY-RIDEAL
      IF(RT.EQ.'ER') RTYPE(J) = 10
C  DUST-GRAIN INTERACTION: cation-grain recombination
      IF(RT.EQ.'CG') RTYPE(J) = 11
C  DUST-GRAIN INTERACTION: SPUTTERING
      IF(RT.EQ.'SP') RTYPE(J) = 12
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: PHOTOREACTION
      IF(RT.EQ.'GP') RTYPE(J) = 17
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: COSMIC-RAY INDUCED PHOTOREACTION
      IF(RT.EQ.'GC') RTYPE(J) = 16
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: REACTIVE DESORPTION linked to LH
      IF(RT.EQ.'RL') RTYPE(J) = 18
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: REACTIVE DESORPTION linked to ER
      IF(RT.EQ.'RE') RTYPE(J) = 19
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: GRAIN ELECTRON CAPTURE
      IF(RT.EQ.'GE') RTYPE(J) = 14
C  DUST-GRAIN INTERACTION: VOLATILE TO REFRACTORY
      IF(RT.EQ.'VR'.AND.RE2.EQ.'PHOTON') RTYPE(J) = 15

      
C  DUST-GRAIN INTERACTION: PHOTODESORPTION - IP 
      IF(RT.EQ.'PD'.AND.RE2.EQ.'INPHOTON') RTYPE(J) = 23
C  DUST-GRAIN INTERACTION: PHOTODESORPTION - AP
      IF(RT.EQ.'PD'.AND.RE2.EQ.'ACPHOTON') RTYPE(J) = 24

C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: PHOTOREACTION - IP - CROSS-SECTIONS NOT AVAILABLE
      IF(RT.EQ.'GP'.AND.RE2.EQ.'INPHOTON') RTYPE(J) = 26
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: PHOTOREACTION - IP - CROSS-SECTIONS AVAILABLE
      IF(RT.EQ.'GP'.AND.RE2.EQ.'INPHOTON'.AND.
     *    STAR.EQ.'CWLeo') RTYPE(J) = 27
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: PHOTOREACTION - IP - CROSS-SECTIONS AVAILABLE - IONISATION RATES
      IF(RT.EQ.'GP'.AND.RE2.EQ.'INPHOTON'.AND.
     *     STAR.EQ.'Millar 2018') RTYPE(J) = 27
      
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: PHOTOREACTION - AP - CROSS-SECTIONS NOT AVAILABLE
      IF(RT.EQ.'GP'.AND.RE2.EQ.'ACPHOTON') RTYPE(J) = 28
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: PHOTOREACTION - AP - CROSS-SECTIONS AVAILABLE
      IF(RT.EQ.'GP'.AND.RE2.EQ.'ACPHOTON'.AND.
     *    STAR.EQ.'CWLeo') RTYPE(J) = 29
C  DUST-GRAIN INTERACTION: GRAIN-SURFACE REACTION: PHOTOREACTION - AP - CROSS-SECTIONS not AVAILABLE - IONISATION RATES
      IF(RT.EQ.'GP'.AND.RE2.EQ.'ACPHOTON'.AND.
     *    STAR.EQ.'Millar 2018') RTYPE(J) = 30

     
C  DUST-GRAIN INTERACTION: VOLATILE TO REFRACTORY - IP
      IF(RT.EQ.'VR'.AND.RE2.EQ.'INPHOTON') RTYPE(J) = 31
C  DUST-GRAIN INTERACTION: VOLATILE TO REFRACTORY - AP
      IF(RT.EQ.'VR'.AND.RE2.EQ.'ACPHOTON') RTYPE(J) = 32

C  DETERMINE INDEX OF CO PHOTODISSOCIATION REACTION
      IF(RE1.EQ.'CO'.AND.RE2.EQ.'PHOTON') ICOR = J
C IDENTIFY REACTION NUMBER OF CO + INPHOTON
      IF(RE1.EQ.'CO'.AND.RE2.EQ.'INPHOTON') ICOIP = J
C IDENTIFY REACTION NUMBER OF CO + ACPHOTON
      IF(RE1.EQ.'CO'.AND.RE2.EQ.'ACPHOTON') ICOAP = J

C  WRITE FORMATTED OUTPUT (ONLY WRITES DATA FOR FIRST TEMP. RANGE)
      WRITE(10,1) I,RE1,RE2,PR1,PR2,PR3,PR4,ALF(J,1),BET(J,1),GAM(J,1)
 1    FORMAT(I4,1X,A12,A12,A12,A12,A5,A5,1PE8.2,0PF7.1,0PF10.1)

      GO TO 100
 300  CONTINUE

      NREAC = J - 1

      WRITE(*,*) 'Reactions read: ',NREAC

C  CATCH BASIC RATE FILE ERROR
      IF(NREAC.LT.1) CALL DIE
     *   ("No reactions defined -- check rate file                    ")

      CLOSE(10)
      CLOSE(URATES)

      RETURN
      END

    
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION TO READ BINDING FILE AND STORE REQUIRED DATA  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE READB(FBIND,UBIND)
C  FBIND = BINDING ENERGY FILE NAME
C  UBIND = BINDING ENERGY FILE UNIT NUMBER

      CHARACTER*500 FBIND,SDATA,TOKEN(1000)
      CHARACTER*12 BSPECIES(1000)
      CHARACTER*1 SEPC,SEP,DUMMY
      DOUBLE PRECISION EBINDS(1000),EDIFF_EBIND
      INTEGER I,UBIND,NBIND
      
C  COMMON BLOCK FOR REACTION DATA STORAGE
      COMMON/BIND/ BSPECIES,EBINDS,NBIND,EDIFF_EBIND

C  BSPECIES = SPECIES WITH BINDING ENERGY
C  EBINDS = CORRESPONING BINDING ENERGIES
C  NBIND = TOTAL NUMBER OF BINDING ENERGIES
C  EDIFF_EBIND = EDIFF/EBIND (INPUT)
      
C  TOKEN (DATA SUBSTRING SEPARATOR)
      SEPC = ':'
      SEP = SEPC

C  OPEN RATE FILE
      OPEN(UNIT=UBIND, FILE=FBIND, STATUS='OLD')

C  DATA FILE READ LOOP
c      J = 0
c      I = 0
c100   J = J+1
c      READ(UBIND,101,END=300) SDATA
c101   FORMAT(A)
!       WRITE(*,*) SDATA

      J=0
      
202   READ(UBIND,*,END=203) DUMMY
      
      IF (DUMMY.EQ."#") J=J+1
      
      GO TO 202
      
203   CONTINUE
      
      CLOSE(UBIND)
      
      OPEN(UNIT=UBIND, FILE=FBIND, STATUS='OLD')
      DO I=1,J      
            READ(UBIND,*)      
      END DO

102   READ(UBIND,*,END=103) BSPECIES(I), EBINDS(I)

      I=I+1
      
      GO TO 102
      
103   CONTINUE

      NBIND = I-1
   
      CLOSE(UBIND)

      RETURN
      END
      
      
