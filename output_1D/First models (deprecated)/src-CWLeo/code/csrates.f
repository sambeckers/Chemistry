CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  SUBROUTINE TO CALCULATE THE RATE COEFFICIENTS (K)  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE RATES(RADIUS)
      DOUBLE PRECISION Y(1000),K(15000),ALF(15000,5),BET(15000,5),
     *   GAM(15000,5),TLOWER(15000,5),TUPPER(15000,5),TOTAL(10),X(10),
     *   RAD,AUV,H2COL,TEMP,ZETA,ALBEDO,GAMCO,ACCR,HNR,X_G,A_G,MLOSS,
     *   RADIUS,PI,MH,MU,KB,DTMIN,DT,AUV_AV,AVEFF,C,HNRMEAN,H2COLEFF,V
      DOUBLE PRECISION GETHNR,GETTEM,GETH2,GETAUV,GETRAD,GETCOR,EARG,
     *   GETTEMDUST,GETDIST,GETMOD,GETETH,GETZP,GETMP
      DOUBLE PRECISION FVOL,LSTAR,GETAUVEFF,AUVEFF,FIC,GSTAR,RSCALE,R,
     *   DELTA_AUV,DIST,BFILL,CFILL,R_STAR,T_STAR,EPSIL,RDUST,NU,
     *   TEMP_DUST,EDIFF_EBIND,R_EPSIL,S_DUST,EBINDS(1000),EBIND,
     *   NACT,SIGMAGRAIN,TOTSITES,NS,DENSITES,NMONO,EBINDA,KAPPAAB,
     *   EBINDB,T_DUST,FUV,ALL_ICE,EDIFFA,EDIFFB,KDIFFA,KDIFFB,KHOPA,
     *   KHOPB,ECHARGE,KERG,DTG,GRAINEXP,AMAX,AMIN,RHODUST,CTEGRAIN,
     *   ND,VDRIFT,S,VTHERMAL,ETH,ECOL,GAMMA,KACCR,ALPHA,IAP,EPT,SN,
     *   ZP,ZT,SN1,MT,MP,ECOL_EV,EMASS,RATEVR,RADVR,AGMEAN,BRANCH,
     *   INCREASEBIND,GE0,HNRMEANDUST,HNRDUST,H2COLDUST,H2COLDUSTEFF,
     *   DELTA_H2COLEFF,AUVDUST,GBIN,RBIN,TBIN,DUMMY
      INTEGER RTYPE(15000),NTR(15000),NREAC,ICO,ICOR,TR,I,J,ICLUMP,
     *   IDGCHEM,ILH,IER,NBIND,Z,NSPEC,ISCALEAUV,IGRAINPHOTON,
     *   ISPUTTERINGWOITKE,ISPUTTERINGTIELENS,SPUTCO,SPUTN2,ICOIP,
     *   SPUTFACTORTWO,IGRAIN0,ISTELLAR,IBIN
      CHARACTER*500 CLUMPMODE,TEMPMODE,REACT1(15000),REACT2(15000)
      CHARACTER*12 BSPECIES(1000),DUMMYRT,DUMMYBS,SPECI(1000)

      COMMON/BL1/ K,X,TOTAL,ACCR,HNR
      COMMON/BL2/ MLOSS,V
      COMMON/BL3/ Y,X_G,A_G,ZETA,ALBEDO,GAMCO,AUV_AV,ICO,GSTAR,RSCALE,
     *   RDUST,DELTA_AUV,DTG,GRAINEXP,AMAX,AMIN,RHODUST,VDRIFT,AGMEAN,
     *   GBIN,ISTELLAR,IBIN,RBIN,TBIN
      COMMON/BL4/ ALF,BET,GAM,TLOWER,TUPPER,RTYPE,NTR,NREAC,ICOR,ICOIP,
     *   ICOAP
      COMMON/BLC/ PI,MH,MU,KB
      COMMON/CLM/ CLUMPMODE,ICLUMP,FVOL,LSTAR,FIC
      COMMON/BLTEMP/ R_STAR,T_STAR,EPSIL,TEMPMODE,R_EPSIL,S_DUST,
     *   T_DUST
      COMMON/ICE/ DENSITES,NMONO,SIGMAGRAIN,NS,TOTSITES,NACT,ALL_ICE
      COMMON/BIND/ BSPECIES,EBINDS,NBIND,NU,EDIFF_EBIND
      COMMON/SPEC/ REACT1,REACT2
      COMMON/CALCICE/ NSPEC,SPECI
      COMMON/CHARGE/ ECHARGE,KERG
      COMMON/SWITCH/ IDGCHEM,ILH,IER,IGRAINSIZEDIST,ISCALEAUV,
     *    ISPUTTERINGWOITKE,ISPUTTERINGTIELENS,SPUTTIELENS,SPUTCO,
     *    SPUTN2,SPUTFACTORTWO,RATEVR,BRANCH,IGRAINPHOTON
      COMMON/INC/ INCREASEBIND
      
      
c       write(*,*) IDGCHEM
! C  UNSHIELDED PHOTODISSOCIATION RATE OF CO
      GE0 = 2.4E-10
      
      
C  H2 NUMBER DENSITY
      HNRMEAN = GETHNR(RADIUS)
!       HNRMEAN = GETHNR(IRUN)
      IF (CLUMPMODE.EQ.'POROSITY'.AND.ICLUMP.EQ.0) THEN
            HNR = FIC*HNRMEAN
      ELSE IF (CLUMPMODE.EQ.'POROSITY'.AND.ICLUMP.EQ.1) THEN	
            HNR = (HNRMEAN/FVOL)*(1-(1-FVOL)*FIC)
      ELSE 
            HNR = HNRMEAN
      END IF
      
      HNRMEANDUST = GETHNR(RDUST)
      IF (CLUMPMODE.EQ.'POROSITY'.AND.ICLUMP.EQ.0) THEN
            HNRDUST = FIC*HNRMEANDUST
      ELSE IF (CLUMPMODE.EQ.'POROSITY'.AND.ICLUMP.EQ.1) THEN	
            HNRDUST = (HNRMEANDUST/FVOL)*(1-(1-FVOL)*FIC)
      ELSE 
            HNRDUST = HNRMEANDUST
      END IF 

C  TEMPERATURE
      TEMP = GETTEM(RADIUS)
C  DUST TEMPERATURE
      TEMP_DUST = GETTEMDUST(RADIUS)

C  H2 COLUMN DENSITY
      H2COL = GETH2(RADIUS,HNRMEAN)
      H2COLDUST = GETH2(RDUST,HNRMEANDUST)
      H2COLEFF = GETH2(RADIUS,HNR)
      H2COLDUSTEFF = GETH2(RDUST,HNRDUST)
      DELTA_H2COLEFF = H2COLDUSTEFF - H2COLEFF

C  H-ATOM ACCRETION RATE
      IF (IDGCHEM.EQ.0) THEN
      ACCR = 0.3*PI*(A_G**2.0)*HNR*X_G*(8.0*KB*TEMP/(PI*MH))**0.5
      ELSE IF (IDGCHEM.EQ.1) THEN
        ACCR = 0.0
      END IF

C  RADIAL UV EXTINCTION
      AUV = GETAUV(H2COL,RADIUS)
      AUVDUST = GETAUV(H2COLDUST,RDUST)
            
C  RADIATION FIELD STRENGTH
      RAD = GETRAD(AUV,RADIUS)

C  define the effective UV extinction from rdust to radius = distance 
      DIST = GETDIST(RADIUS)
      DELTA_AUV = AUVDUST - AUV

C UV FLUX = INTERSTELLAR RADIATION FIELD + SECONDARY UV PHOTONS (COSMIC RAYS)      
      FUV = (1E8)*RAD+1e4

C  CALCULATE TOTAL ICE NUMBER DENSITY
      ALL_ICE = 0.0
      DO Z=1,NSPEC
        IF (SPECI(Z)(1:1).EQ.'G') THEN 
        ALL_ICE = ALL_ICE + (Y(Z)*HNR)
        END IF
      END DO

      
      
C  SET DIFFERENT DUST GRAIN PARAMETERS
C  IF NO GSD: USE CONSTANT DUST GRAINS
C  ELSE, USE GSD PARAMETERS (SEE PAPER I, PAPER II)
      IF (IGRAINSIZEDIST.EQ.0) THEN
      ND = X_G*HNR
      TOTSITES = NS*4.*PI*A_G**2
      SIGMAGRAIN = PI*A_G**2*X_G*HNR
      DENSITES = TOTSITES*X_G*HNR
      
      
      ELSE
      
      CTEGRAIN = 3.*(4.+GRAINEXP)*DTG*MU*MH*HNR /
     *    (4.*PI*RHODUST*(AMAX**(4.+GRAINEXP)-AMIN**(4.+GRAINEXP)))
     
      ND = CTEGRAIN/(1.+GRAINEXP) *
     *     (AMAX**(1.+GRAINEXP)-AMIN**(1.+GRAINEXP))

      SIGMAGRAIN = PI*CTEGRAIN/(3.+GRAINEXP) *
     *     (AMAX**(3.+GRAINEXP)-AMIN**(3.+GRAINEXP))
      
      DENSITES = CTEGRAIN*NS*4.*PI/(3.+GRAINEXP) *
     *     (AMAX**(3.+GRAINEXP)-AMIN**(3.+GRAINEXP))
      
      TOTSITES = DENSITES/ND
      
!       AGMEAN = SQRT(SIGMAGRAIN/PI/ND)
      
      END IF
      
C  CALCULATE NUMBER OF MONOLAYERS      
      NMONO = ALL_ICE/DENSITES
      
      
C  RADIATION FIELD FOR V->R REACTION (ISRF + DUST SHIELDING + EXTINCTION DUE TO WATER - MAIN ICE COMPONENT)      
      RADVR = RAD*EARG(GAMCO-2.2*(AUV/AUV_AV))
      
      
      
      
c       write(24,*) AUV,AUVDUST,HNRMEAN,HNRMEANDUST,TEMP,TEMP_DUST,
c      *          H2COL,H2COLDUST,H2COLEFF,H2COLDUSTEFF,RAD,DELTA_AUV,
c      *          CTEGRAIN,RADVR
      
C  *****CALCULATE REACTION RATES****************************************

      DO J=1,NREAC

C  DETERMINE TEMPERATURE RANGE FOR RATE COEFFICIENTS

C  INITIALISE TEMPERATURE RANGE INDEX
         TR = 1
C  IF MULTIPLE TEMP RANGES, FIND THE CLOSEST MATCH TO CURRENT TEMP.
         IF(NTR(J).GT.1) THEN
            DTMIN = 1.0E20
            DO I = 1,NTR(J)
               IF(TEMP.GE.TLOWER(J,I).AND.TEMP.LT.TUPPER(J,I)) THEN
                  TR = I
                  GO TO 201
               ELSE
                  DT = ABS(TLOWER(J,I)-TEMP)
                  IF(ABS(TUPPER(J,I)-TEMP).LT.DT) THEN
                     DT = ABS(TUPPER(J,I)-TEMP)
                  END IF
                  IF(DT.LT.DTMIN) THEN
                     DTMIN = DT
                     TR = I
                  END IF
               END IF
            END DO
         END IF

         
C  READ IN BINDING ENERGY FOR THERMAL DESORPTION         
      IF (RTYPE(J).EQ.7.OR.RTYPE(J).EQ.12) THEN
            DUMMYRT = TRIM(REACT1(J))
                              
            DO Z=1,NBIND          
            DUMMYBS = TRIM(BSPECIES(Z))
            IF (DUMMYRT(2:).EQ.DUMMYBS) THEN
                  EBINDA = EBINDS(Z)
                  CONTINUE
            END IF  
            
            END DO
            
            IF (NMONO.LT.2) THEN
                EBINDA = EBINDA + EBINDA*INCREASEBIND
            END IF
        
      END IF   
         
C  READ IN BINDING ENERGY FOR BOTH REACTANTS FOR GRAIN SURFACE REACT
      IF (RTYPE(J).EQ.9.OR.RTYPE(J).EQ.18) THEN
            DUMMYRT = TRIM(REACT1(J))
            DO Z=1,NBIND          
            DUMMYBS = TRIM(BSPECIES(Z))
            IF (DUMMYRT(2:).EQ.DUMMYBS) THEN
                  EBINDA = EBINDS(Z)
                  CONTINUE
            END IF  
            
            END DO
            
            DUMMYRT = TRIM(REACT2(J))

            DO Z=1,NBIND          
            DUMMYBS = TRIM(BSPECIES(Z))
            IF (DUMMYRT(2:).EQ.DUMMYBS) THEN
                  EBINDB = EBINDS(Z)
                  CONTINUE
            END IF  
            
            END DO      
            
            IF (NMONO.LT.2) THEN
                EBINDA = EBINDA + EBINDA*INCREASEBIND
                EBINDB = EBINDB + EBINDB*INCREASEBIND
            END IF
      END IF   
         
         
         
         

C  COMPUTE RATE COEFFICIENTS BASED ON REACTION TYPE
C  CLASSIC REACTIONS
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C  COSMIC RAY PARTICLE RATE
 201  IF(RTYPE(J).EQ.1) THEN
      K(J) = ALF(J,TR)*ZETA

C  COSMIC RAY PHOTO-RATE
      ELSE IF(RTYPE(J).EQ.2) THEN
      K(J) = ZETA*ALF(J,TR)*((TEMP/300.0)**BET(J,TR))*GAM(J,TR)
     *   /(1.0-ALBEDO)

C  PHOTO-REACTION RATE 
      ELSE IF ((RTYPE(J).EQ.3)) THEN
c       DUMMY = (GAMCO-GAM(J,TR))*(AUV/AUV_AV)
c       IF (DUMMY.LT.710) THEN
      
      K(J) = RAD*ALF(J,TR)*EARG((GAMCO-GAM(J,TR))*(AUV/AUV_AV))
      
c       ELSE
c       
c       K(J) = RAD*ALF(J,TR)*8E300
c       END IF
c       
c       IF (J.EQ.6217) THEN
c       WRITE(*,*) K(J),RAD,AUV,
c      *     (GAMCO-GAM(J,TR))*(AUV/AUV_AV),
c      *     EARG((GAMCO-GAM(J,TR))*(AUV/AUV_AV))
c       END IF

      
CCC  GAS-PHASE INTERNAL PHOTON REACTIONS      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

c calculate internal stellar photon rates at RSCALE (50 R*) - scale to integrated flux relative to ISM value
      ELSE IF(RTYPE(J).EQ.4) THEN
      K(J) = GSTAR*((RSCALE/DIST)**2)*ALF(J,TR)*EARG(-GAM(J,TR)* 
     *  (DELTA_AUV/AUV_AV))
c set K(J) = 0 to exclude IP
      IF(ISTELLAR.EQ.0) K(J) = 0.0


c calculate internal stellar photon rates at RSCALE (50 R*) - using detailed cross-sections
      ELSE IF(RTYPE(J).EQ.5) THEN
      K(J) = ((RSCALE/DIST)**2)*ALF(J,TR)*EARG(-GAM(J,TR)*
     *  (DELTA_AUV/AUV_AV))
c set K(J) = 0 to exclude inner stellar PHOTONS
      IF (ISTELLAR.EQ.0) K(J) = 0.0

      
      
c calculate internal BINARY COMPANION photon rates at RSCALE (50 RSTAR) - scale to integrated flux relative to ISM value
c (2./pi) scales the rate to the average disk surface area seen by a point at an angle theta
C average = 2r^2, not pi*r^2
      ELSE IF(RTYPE(J).EQ.20) THEN
      K(J) = GBIN*((RSCALE/DIST)**2)*ALF(J,TR)*
     *  EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
     
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
      IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
c set K(J) = 0 to exclude companion photons
      IF(IBIN.EQ.0) K(J) = 0.0
c

c calculate internal BINARY COMPANION photon rates at RSCALE (50 RSTAR) - using detailed cross-sections
c (2./pi) scales the rate to the average disk surface area seen by a point at an angle theta
C average = 2r^2, not pi*r^2
      ELSE IF(RTYPE(J).EQ.21) THEN
      K(J) = (((RBIN/R_STAR)**2)*(RSCALE/DIST)**2)*ALF(J,TR)*
     *  EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
     
c       IF (J.EQ.11352) THEN
c       WRITE(*,*) -GAM(J,TR)*(DELTA_AUV/AUV_AV),
c      *  EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV)) 
c       END IF

C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
      IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
c set K(J) = 0 to exclude BINARY/DISK COMPANION photons
      IF(IBIN.EQ.0) K(J) = 0.0
C
        
        
c calculate internal BINARY COMPANION photoionisation rates at RSCALE (50 RSTAR) - scale to integrated flux relative to ISM value
c (2./pi) scales the rate to the average disk surface area seen by a point at an angle theta
C average = 2r^2, not pi*r^2
      ELSE IF(RTYPE(J).EQ.22) THEN
         IF (TBIN.EQ.4000.) THEN
             K(J) = ((RBIN/R_STAR)**2)*((RSCALE/DIST)**2)*ALF(J,TR)*
     *         EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
            IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
c set K(J) = 0 to exclude BINARY/DISK COMPANION photons
             IF(IBIN.EQ.0) K(J) = 0.0
C
         ELSE IF (TBIN.EQ.6000.) THEN
            K(J) = ((RSCALE/DIST)**2)*ALF(J,TR)*
     *        EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
            IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
c set K(J) = 0 to exclude BINARY/DISK COMPANION photons
            IF(IBIN.EQ.0) K(J) = 0.0
C
         ELSE IF (TBIN.EQ.10000.) THEN
             K(J) = GBIN*((RSCALE/DIST)**2)*ALF(J,TR)*
     *         EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
            IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
c set K(J) = 0 to exclude companion photons
            IF(IBIN.EQ.0) K(J) = 0.0
          END IF
      

CCC  DUST-GAS INTERACTIONS
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      

C  FREEZE-OUT/ACCRETION
C  ALPHA = molecular mass (not used), BETA = sticking coefficient, GAMMA = /
      ELSE IF (RTYPE(J).EQ.6) THEN
      IF (IDGCHEM.EQ.1) THEN
!         K(J) = BET(J,TR)*(8.0*KB*TEMP/(PI*ALF(J,TR)*MH))**0.5
!      *    *SIGMAGRAIN
        VTHERMAL = (8.0*KB*TEMP/(PI*ALF(J,TR)*MH))**0.5
        K(J) = BET(J,TR)*VTHERMAL*SIGMAGRAIN
        IF (VDRIFT.NE.0.0) THEN
          S = VDRIFT/VTHERMAL
          K(J) = K(J)*GETMOD(S)
        END IF
          
      ELSE
        K(J) = 0.0
      END IF

      
C  THERMAL DESORPTION
C  ALPHA = molecular mass (not used), BETA = sticking coefficient (not used), GAMMA = /
      ELSE IF (RTYPE(J).EQ.7) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (NMONO.LT.2) THEN
! C        1.36e-16 converts energy from K to erg = cm2 g / s2
!           K(J) = (2.*NS*EBINDA*1.36e-16/(PI*PI*ALF(J,TR)*MH))**0.5*
!      *      EXP(-EBINDA/TEMP_DUST)
          K(J) = NU*EXP(-EBINDA/TEMP_DUST)
        ELSE
!           K(J) = (2.*NS*EBINDA*1.36e-16/(PI*PI*ALF(J,TR)*MH))**0.5*
!      *      EXP(-EBINDA/(TEMP_DUST))*NACT*DENSITES/(ALL_ICE)
          K(J) = NU*EXP(-EBINDA/(TEMP_DUST))*NACT*DENSITES/(ALL_ICE)
        END IF  
        
c         IF (J.EQ.7441) THEN
c         write(*,*) EBINDA, INCREASEBIND,K(J)
c         END IF 
      ELSE
            K(J) = 0.0
      END IF
    

C  PHOTODESORPTION - ISRF
C  ALPHA = /, BETA = yield, GAMMA = gamma of photoreactions
      ELSE IF (RTYPE(J).EQ.8) THEN
      IF (IDGCHEM.EQ.1) THEN
        FUV = (1E8)*RAD*EARG((GAMCO-GAM(J,TR))*(AUV/AUV_AV))+1e4
        IF (ALL_ICE.EQ.0.0) THEN
        K(J) = 0.0
        ELSE
          IF (NMONO.LT.2) THEN
            K(J) = BET(J,TR)*ALF(J,TR)*FUV/(NS*4.*NACT)
  !           K(J) = BET(J,TR)*ALF(J,TR)*FUV*SIGMAGRAIN/(NACT*DENSITES)
          ELSE
              ! FIRST ORDER RATE
            K(J) = BET(J,TR)*ALF(J,TR)*FUV*SIGMAGRAIN/ALL_ICE
          END IF 
        END IF  
      ELSE
        K(J) = 0.0
      END IF
    

C  PHOTODESORPTION - Internal photons IP  
      ELSE IF (RTYPE(J).EQ.23) THEN
      IF (IDGCHEM.EQ.1.AND.ISTELLAR.EQ.1) THEN
        FUV = (1E8)*GSTAR*((RSCALE/DIST)**2)*
     *     EARG((-GAM(J,TR))*(DELTA_AUV/AUV_AV))
        IF (ALL_ICE.EQ.0.0) THEN
        K(J) = 0.0
        ELSE
          IF (NMONO.LT.2) THEN
            K(J) = BET(J,TR)*ALF(J,TR)*FUV/(NS*4.*NACT)
          ELSE
            K(J) = BET(J,TR)*ALF(J,TR)*FUV*SIGMAGRAIN/ALL_ICE
          END IF 
        END IF  
c set K(J) = 0 to exclude IP
      IF(ISTELLAR.EQ.0) K(J) = 0.0
      ELSE
        K(J) = 0.0
      END IF
      
C  PHOTODESORPTION - Companion photons AP  
      ELSE IF (RTYPE(J).EQ.24) THEN
      IF (IDGCHEM.EQ.1) THEN
        FUV = (1E8)*GBIN*((RSCALE/DIST)**2)*
     *     EARG((-GAM(J,TR))*(DELTA_AUV/AUV_AV))
        IF (ALL_ICE.EQ.0.0) THEN
        K(J) = 0.0
        ELSE
          IF (NMONO.LT.2) THEN
            K(J) = BET(J,TR)*ALF(J,TR)*FUV/(NS*4.*NACT)
          ELSE
            K(J) = BET(J,TR)*ALF(J,TR)*FUV*SIGMAGRAIN/ALL_ICE
          END IF 
        END IF
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
      IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
c set K(J) = 0 to exclude BINARY/DISK COMPANION photons
      IF(IBIN.EQ.0) K(J) = 0.0
      ELSE
        K(J) = 0.0
      END IF

    
    
    
    
    
CCC  GRAIN SURFACE CHEMISTRY
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C   LANGUIR-HINSHELWOOD
C  ALPHA = reaction barrier, BETA = branching ratio, GAMMA = /
      ELSE IF (RTYPE(J).EQ.9.OR.RTYPE(J).EQ.18) THEN
      IF (ILH.EQ.1) THEN
        EDIFFA = EDIFF_EBIND*EBINDA
        EDIFFB = EDIFF_EBIND*EBINDB
      
        KHOPA = NU*EXP(-EDIFFA/TEMP_DUST)
        KHOPB = NU*EXP(-EDIFFB/TEMP_DUST)
        
        KDIFFA = KHOPA/(TOTSITES)
        KDIFFB = KHOPB/(TOTSITES)
      
        KAPPAAB = EXP(-ALF(J,TR)/TEMP_DUST)
      
        IF (NMONO.LT.2) THEN
          K(J) = BET(J,TR)*KAPPAAB*(KDIFFA+KDIFFB)/(ND)
        ELSE
          K(J) = BET(J,TR)*(KAPPAAB*(KDIFFA+KDIFFB)/(ND))
     *             *(NACT*DENSITES/(ALL_ICE))**2.0
        END IF  
        
C   REACTIVE DESORPTION        
        IF (RTYPE(J).EQ.18) THEN
          K(J) = K(J) * BRANCH
        END IF
        
      ELSE
         K(J) = 0.0
      END IF
      
C   ELEY-RIDEAL
C   ALPHA = reaction barrier, BETA = mass gas-phase reactant, GAMMA = /
      ELSE IF (RTYPE(J).EQ.10.OR.RTYPE(J).EQ.19) THEN
      IF (IER.EQ.1) THEN
        IF (ALL_ICE.EQ.0.0) THEN
        K(J) = 0.0
        
        ELSE
        
        KAPPAAB = EXP(-ALF(J,TR)/TEMP_DUST)
        
        IF (NMONO.LT.2) THEN
          K(J) = KAPPAAB*(8.0*KB*TEMP/(PI*BET(J,TR)*MH))**0.5
     *      *SIGMAGRAIN/(NACT*DENSITES)        
        ELSE
          K(J) = KAPPAAB*(8.0*KB*TEMP/(PI*BET(J,TR)*MH))**0.5
     *      *SIGMAGRAIN/ALL_ICE
        END IF

C   REACTIVE DESORPTION        
        IF (RTYPE(J).EQ.19) THEN
          K(J) = K(J) * BRANCH
        END IF
        
        END IF
      ELSE
         K(J) = 0.0
      END IF

      
            
      
      
      
C   CATION-GRAIN RECOMBINATION --- WITHOUT EXPLICIT TREATMENT GRAIN
C   ALPHA = branching ratio, BETA = /, GAMMA = mass gas-phase reactant
      ELSE IF (RTYPE(J).EQ.11) THEN
      IF (IDGCHEM.EQ.1) THEN
        K(J) = ALF(J,TR)*SIGMAGRAIN
     *    *SQRT(8.0*KB*TEMP/(PI*MH*GAM(J,TR)))
     *    *(1.0 + (ECHARGE**2.0/(AGMEAN*KERG*TEMP)))
     *      *(1.0 + SQRT(2.0*ECHARGE**2.0
     *      /((AGMEAN*KERG*TEMP) + 2.0*ECHARGE**2.0)))*ND
      ELSE
        K(J) = 0.0
      END IF
      
      
C   GRAIN ELECTRON CAPTURE -- PLACEHOLDER -- ACTUALLY NOT INCLUDED
C  ALPHA = sticking coefficient, BETA = /, GAMMA = /
      ELSE IF (RTYPE(J).EQ.14) THEN
      IF (IDGCHEM.EQ.1) THEN
        VTHERMAL = SQRT(8.0*KB*TEMP/(PI*EMASS*MH))
        K(J) = ALF(J,TR)*VTHERMAL*SIGMAGRAIN
        IF (VDRIFT.NE.0.0) THEN
          S = VDRIFT/VTHERMAL
          K(J) = K(J)*GETMOD(S)
        END IF
      ELSE
        K(J) = 0.0
      END IF
      
      

      
C  GRAIN-SURFACE REACTION: COSMIC-RAY INDUCED PHOTOREACTION
      ELSE IF(RTYPE(J).EQ.16) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (IGRAINPHOTON.EQ.1) THEN
          K(J) = ZETA*ALF(J,TR)*((TEMP/300.0)**BET(J,TR))*GAM(J,TR)
     *      /(1.0-ALBEDO)
        ELSE
          K(J) = 0.0
        END IF
      ELSE
        K(J) = 0.0
      END IF

      
      

C  GRAIN-SURFACE REACTION: PHOTOREACTION - ISRF
      ELSE IF ((RTYPE(J).EQ.17)) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (IGRAINPHOTON.EQ.1) THEN
          K(J) = RAD*ALF(J,TR)*EARG((GAMCO-GAM(J,TR))*(AUV/AUV_AV))
        ELSE
          K(J) = 0.0
        END IF
      ELSE
        K(J) = 0.0
      END IF
 
 
 
 
      
C  GRAIN-SURFACE REACTION: PHOTOREACTION IP  
C Cross sections not available
      ELSE IF ((RTYPE(J).EQ.26.AND.ISTELLAR.EQ.1)) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (IGRAINPHOTON.EQ.1) THEN
          K(J) = GSTAR*((RSCALE/DIST)**2)*ALF(J,TR)*EARG(-GAM(J,TR)* 
     *      (DELTA_AUV/AUV_AV))
        ELSE
          K(J) = 0.0
        END IF
      ELSE
        K(J) = 0.0
      END IF
      
C  GRAIN-SURFACE REACTION: PHOTOREACTION IP  
C Cross sections available
      ELSE IF ((RTYPE(J).EQ.27.AND.ISTELLAR.EQ.1)) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (IGRAINPHOTON.EQ.1) THEN
          K(J) = ((RSCALE/DIST)**2)*ALF(J,TR)*EARG(-GAM(J,TR)*
     *     (DELTA_AUV/AUV_AV))
        ELSE
          K(J) = 0.0
        END IF
      ELSE
        K(J) = 0.0
      END IF

      
      
      
C  GRAIN-SURFACE REACTION: PHOTOREACTION AP  
C Cross sections not available
      ELSE IF ((RTYPE(J).EQ.28)) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (IGRAINPHOTON.EQ.1) THEN
          K(J) = GBIN*((RSCALE/DIST)**2)*ALF(J,TR)*
     *      EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
        ELSE
          K(J) = 0.0
        END IF
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
      IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
      IF(IBIN.EQ.0) K(J) = 0.0
      ELSE
        K(J) = 0.0
      END IF


C  GRAIN-SURFACE REACTION: PHOTOREACTION AP  
C Cross sections available
      ELSE IF ((RTYPE(J).EQ.29)) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (IGRAINPHOTON.EQ.1) THEN
          K(J) = (((RBIN/R_STAR)**2)*(RSCALE/DIST)**2)*ALF(J,TR)*
     *      EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
        ELSE
          K(J) = 0.0
        END IF
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
      IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
      IF(IBIN.EQ.0) K(J) = 0.0
      ELSE
        K(J) = 0.0
      END IF
      
      
C  GRAIN-SURFACE REACTION: PHOTOREACTION AP  
C Cross sections not available, scaling photoionisation rate
      ELSE IF ((RTYPE(J).EQ.30)) THEN
      IF (IDGCHEM.EQ.1) THEN
        IF (IGRAINPHOTON.EQ.1) THEN
        
          IF (TBIN.EQ.4000.) THEN
             K(J) = ((RBIN/R_STAR)**2)*((RSCALE/DIST)**2)*ALF(J,TR)*
     *         EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))

          ELSE IF (TBIN.EQ.6000.) THEN
            K(J) = ((RSCALE/DIST)**2)*ALF(J,TR)*
     *        EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))

          ELSE IF (TBIN.EQ.10000.) THEN
             K(J) = GBIN*((RSCALE/DIST)**2)*ALF(J,TR)*
     *         EARG(-GAM(J,TR)*(DELTA_AUV/AUV_AV))
          END IF        
        
        
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
          IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
          IF(IBIN.EQ.0) K(J) = 0.0
     
        ELSE
          K(J) = 0.0
        END IF
      ELSE
        K(J) = 0.0
      END IF
      
      

      
      
C   VOLATILE TO REFRACTORY - ISRF
C  ALPHA = switch for all or complex ices (ratesfile), BETA = /, GAMMA = /
      ELSE IF (RTYPE(J).EQ.15) THEN
      IF (IDGCHEM.EQ.1) THEN
      RADVR = RAD*EARG(GAMCO-2.2*(AUV/AUV_AV))
        IF (RADVR.LT.1e-20) THEN
          K(J) = 0.0
        ELSE
          K(J) = ALF(J,TR)*RATEVR*RADVR
        END IF        
      ELSE
        K(J) = 0.0
      END IF
      
      
C   VOLATILE TO REFRACTORY - IP
      ELSE IF (RTYPE(J).EQ.31) THEN
      IF (IDGCHEM.EQ.1.AND.ISTELLAR.EQ.1) THEN
        RADVR = GSTAR*((RSCALE/DIST)**2)*
     *      EARG(GAMCO-2.2*(DELTA_AUV/AUV_AV))
        IF (RADVR.LT.1e-20) THEN
          K(J) = 0.0
        ELSE
          K(J) = ALF(J,TR)*RATEVR*RADVR
        END IF        
      ELSE
        K(J) = 0.0
      END IF

C   VOLATILE TO REFRACTORY - AP
      ELSE IF (RTYPE(J).EQ.32) THEN
      IF (IDGCHEM.EQ.1) THEN
        RADVR = GBIN*((RSCALE/DIST)**2)*
     *     EARG(GAMCO-2.2*(DELTA_AUV/AUV_AV))
        IF (RADVR.LT.1e-20) THEN
          K(J) = 0.0
        ELSE
          K(J) = ALF(J,TR)*RATEVR*RADVR
        END IF        
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
        IF(IBIN.EQ.1) K(J) = (2.0/PI)*K(J)
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
        IF(IBIN.EQ.0) K(J) = 0.0
      ELSE
        K(J) = 0.0
      END IF
      
      
      
      
      
      
C   SPUTTERING
C   ALPHA = molecular mass ice, BETA = mean molecular mass ice, GAMMA = mean atomic number ice 
      ELSE IF (RTYPE(J).EQ.12) THEN
            
      IF (ISPUTTERINGWOITKE.EQ.1) THEN
      
        IF (SPUTCO.EQ.0.AND.REACT2(J).EQ.'CO') THEN
          K(J) = 0.0
        ELSE IF (SPUTN2.EQ.0.AND.REACT2(J).EQ.'N2') THEN
          K(J) = 0.0
        ELSE IF (SPUTH2O.EQ.0.AND.REACT2(J).EQ.'H2O') THEN
          K(J) = 0.0
        ELSE IF (SPUTC2H2.EQ.0.AND.REACT2(J).EQ.'C2H2') THEN
          K(J) = 0.0
        ELSE IF (SPUTHCN.EQ.0.AND.REACT2(J).EQ.'HCN') THEN
          K(J) = 0.0

        ELSE
          MT = ALF(J,TR) !NORMAL MOLECULAR MASS
          MP = GETMP(REACT2(J))
          
          ETH = GETETH(MT,MP,EBINDA)
          
          ECOL = 0.5*MP*MH*VDRIFT**2.*6.24150913E11

          IF (ECOL.GE.ETH) THEN
          
            VTHERMAL = (8.0*KB*TEMP/(PI*MP*MH))**0.5
            S = VDRIFT/VTHERMAL
            KACCR =  VTHERMAL*SIGMAGRAIN*GETMOD(S)
            GAMMA = 4.*MP*MT/((MP+MT)**2.)
          
            K(J) = 0.0064*MT*GAMMA**(5./3.)*(ECOL/ETH)**0.25
     *      *(1.-(ETH/ECOL))**3.5 * KACCR/HNR 
            
            IF (SPUTFACTORTWO.EQ.1) THEN
              K(J) = K(J)*2.
            END IF

            
          ELSE
            K(J) = 0.0
          END IF
        
        END IF
        
        
      ELSE IF (ISPUTTERINGTIELENS.EQ.1) THEN
      
        IF (SPUTCO.EQ.0.AND.REACT2(J).EQ.'CO') THEN
          K(J) = 0.0
        ELSE IF (SPUTN2.EQ.0.AND.REACT2(J).EQ.'N2') THEN
          K(J) = 0.0
        ELSE IF (SPUTH2O.EQ.0.AND.REACT2(J).EQ.'H2O') THEN
          K(J) = 0.0
        ELSE IF (SPUTC2H2.EQ.0.AND.REACT2(J).EQ.'C2H2') THEN
          K(J) = 0.0
        ELSE IF (SPUTHCN.EQ.0.AND.REACT2(J).EQ.'HCN') THEN
          K(J) = 0.0

        ELSE
        
          MT = BET(J,TR) !MEAN MOLECULAR MASS          
          MP = GETMP(REACT2(J))
          ZP = GETZP(REACT2(J))
          ZT = GAM(J,TR)

          ETH = GETETH(MT,MP,EBINDA) !in eV
          
          ECOL = 0.5*MP*MH*VDRIFT**2. !in erg
          ECOL_EV = ECOL*6.24150913E11 !in eV
          
          
          IF (MT/MP.GT.5) THEN
c             ALPHA = 0.3*(MT/MP)**(2./3.) *((0.01*MT/MP)+1.)**-1.
            ALPHA = 0.3*(MT/MP)**(2./3.) /((0.01*MT/MP)+1.)
          ELSE
            ALPHA = 0.3*(MT/MP)**(2./3.)
          END IF
          
          

          IF (ECOL_EV.GT.ETH) THEN !compare ev energies
            VTHERMAL = (8.0*KB*TEMP/(PI*MP*MH))**0.5
            S = VDRIFT/VTHERMAL
            KACCR =  VTHERMAL*SIGMAGRAIN*GETMOD(S)
            
!           0.529e-8 = Bohr radius in cm
            IAP = 0.885*0.529e-8*(ZP**(2./3.)+ZT**(2./3))**(-0.5)
          
            EPT = MT/(MT+MP) * 
     *           IAP/(ZP*ZT*ECHARGE**2.0)*ECOL
            SN1 = 3.411*(EPT)**0.5*LOG(EPT+2.718) /
     *          (1.+6.35*EPT**0.5+EPT*(-1.708+6.882*EPT**0.5))
            SN = 4.2*PI*IAP*ZT*ZP*ECHARGE**2. *
     *          MP/(MP+MT)*SN1
     
            K(J) = 3.56/(EBINDA*8.621738E-5) * MP/(MP+MT) 
     *         * ZT*ZP /(SQRT(ZT**(2./3.)+ZP**(2./3.))) 
     *         * ALPHA * SN1 * (1.- (ETH/ECOL_EV)**(2./3.)) 
     *         * (1. - ETH/ECOL_EV)**2. *KACCR/HNR    
     
            
            
            IF (SPUTFACTORTWO.EQ.1) THEN
              K(J) = K(J)*2.
            END IF
            
            
            
          ELSE
            K(J) = 0.0
          END IF
          
        END IF
        

      ELSE
        K(J) = 0.0
      END IF
      
      
C  DEFAULT BINARY REACTION RATE
      ELSE
      K(J) = ALF(J,TR)*((TEMP/300.0)**BET(J,TR))*EARG(-GAM(J,TR)/TEMP)
      END IF

      END DO
      

      
C  SET CO PHOTODISSOCIATION RATE FOR EXTERNAL PHOTONS
      K(ICOR) = GETCOR(H2COLEFF,Y(ICO),V,AUV)
   
C  SET CO PHOTODISSOCIATION RATE FOR STELLAR BB PHOTONS
      K(ICOIP) = GETCOR(DELTA_H2COLEFF,Y(ICO),V,DELTA_AUV)/GE0
      K(ICOIP) = K(ICOIP)*((RSCALE/DIST)**2)*ALF(ICOIP,TR)
      IF(ISTELLAR.EQ.0) K(ICOIP) = 0.0     

C  SET CO PHOTODISSOCIATION RATE FOR INTERNAL BINARY COMPANION PHOTONS
      K(ICOAP) = GETCOR(DELTA_H2COLEFF,Y(ICO),V,DELTA_AUV)/GE0
      K(ICOAP) = K(ICOAP)*((RBIN/R_STAR)**2)*((RSCALE/DIST)**2)*
     *  ALF(ICOAP,TR)
C RATE IS RESET IF IBIN = 1, IE A DISK COMPANION
      IF(IBIN.EQ.1) K(ICOAP) = (2.0/PI)*K(ICOAP)
c set K(J) = 0 to exclude BINARY/DISK COMPANION photons
      IF(IBIN.EQ.0) K(ICOAP) = 0.0

c       write(*,*) K(ICOR),K(ICOIP),K(ICOAP)
c       write(*,*) DELTA_H2COLEFF,Y(ICO),V,DELTA_AUV,GE0
c       write(*,*) ICOR,ICOIP,ICOAP
c       write(*,*) H2COLEFF,Y(ICO),V,AUV
c       


      
c       write(20,*) K(ICOR),H2COLEFF,Y(ICO),V,AUV
c 
c       write(17,*) K(11638),GETCOR(DELTA_H2COLEFF,Y(ICO),V,DELTA_AUV),
c      *    (RSCALE/DIST)**2,
c      *    DELTA_H2COLEFF,Y(ICO),V,DELTA_AUV,GE0,
c      *    RSCALE,DIST,ALF(ICOIP,TR)
c 
c       write(18,*) K(ICOAP),GETCOR(DELTA_H2COLEFF,Y(ICO),V,DELTA_AUV),
c      *    (RBIN/R_STAR)**2,
c      *    DELTA_H2COLEFF,Y(ICO),V,DELTA_AUV,GE0,
c      *    RBIN,R_STAR,RSCALE,DIST,ALF(ICOIP,TR)
      
c       WRITE(33,*) RADIUS
c       DO I=1,13670
c       IF(ISNAN(K(I))) THEN
c       WRITE(33,*) I
c       END IF
c       END DO
c       
c       write(*,*) K(7161)
      
      RETURN
      END
