;******************************************************
;
; mjoclivar_14.ncl
;
;***********************************************************
; Combined EOFs
; Latest Update: July, 2016: Eun-Pa Lim; Bureau of Meteorology, Australia
;***********************************************************
;;
;;      The following are automatically loaded from 6.2.0 onward
;;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"  
;;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"  
;;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl" 

undef("read_rename")
function read_rename(f[1]:file, varName[1]:string       \
                    ,iStrt[1]:integer, iLast[1]:integer \
                    ,latS[1]:numeric , latN[1]:numeric  )
; Utility to force specific named dimensions
; This is done for historical reasons (convenience) 
begin
   work    = f->$varName$(iStrt:iLast,{latS:latN},:)   ; (time,lat,lon)
   work!0  = "time"                                    ; CAM model names
   work!1  = "lat"
   work!2  = "lon"
   return(work)
end
; =========================>  MAIN  <==============================
begin
   neof    =  2

   latS    = -15
   latN    =  15

   ymdStrt = 20170101                         ; start yyyymmdd
   ymdLast = 20231231                         ; last  

   yrStrt  = ymdStrt/10000
   yrLast  = ymdLast/10000

   pltDir  = "./"                             ; plot directory
   pltType = "png" 
   pltName = "mjoclivar"   

   diri    = "/glade/campaign/univ/uchi0014/benchmark-dev/MJO/"                             ; input directory

   filolr  = "olr_test_clim_s2s_144x72.nc"
   filu200 = "u250_test_clim_s2s_144x72.nc"
   filu850 = "u850_test_clim_s2s_144x72.nc"

;************************************************
; create BandPass Filter
;************************************************
  ihp      = 2                             ; bpf=>band pass filter
  nWgt     = 201
  sigma    = 1.0                           ; Lanczos sigma
  fca      = 1./100.
  fcb      = 1./20.
  wgt      = filwgts_lanczos (nWgt, ihp, fca, fcb, sigma )

;***********************************************************
; Find the indices corresponding to the start/end times
;***********************************************************
   f       = addfile (diri+filolr , "r")                         
   TIME    = f->time                          ; days since ...

   YMD     = cd_calendar(TIME, -2)            ; entire (time,6)

   iStrt   = ind(YMD.eq.ymdStrt)              ; index start
   iLast   = ind(YMD.eq.ymdLast)              ; index last 
;   delete([/ TIME, YMD /])
;   print(iStrt)
;   print(iLast)
   ;exit()

;***********************************************************
; Read anomalies
;***********************************************************

   work    = read_rename(f,"OLR_clim",iStrt,iLast,latS,latN) ; (time,lat,lon)
   printMinMax(work,0)
   OLR     = dim_avg_n_Wrap(work, 1)                         ; (time,lon)
   printMinMax(OLR, 0)

   f       = addfile (diri+filu850 , "r")                         
   work    = read_rename(f,"u850_clim",iStrt,iLast,latS,latN) ; (time,lat,lon) 
   U850    = dim_avg_n_Wrap(work, 1)          ; (time,lon)
   printMinMax(U850, 0)

   f       = addfile (diri+filu200 , "r")                         
   work    = read_rename(f,"u250_clim",iStrt,iLast,latS,latN) ; (time,lat,lon) 
   U200    = dim_avg_n_Wrap(work, 1)          ; (time,lon)
   printMinMax(U200, 0)

   dimw    = dimsizes( work )
   ntim    = dimw(0)
   nlat    = dimw(1)
   mlon    = dimw(2)
   delete(work)

   lon     = OLR&lon                                          
   time    = OLR&time                         
   date    = cd_calendar(time, -2)            ; yyyymmdd
   print(date(0))


;___________________read predictions

files_pred = systemfunc("ls /glade/campaign/univ/uchi0014/benchmark-dev/MJO/cal_pc/*ens_1.*144x72")
;print(files_pred)
nfiles = dimsizes(files_pred)

do it = 0, nfiles-1

fp = addfile(files_pred(it), "r")
timep = fp->time
YMDp = cd_calendar(timep, -2)
;print(YMDp(0))

print("working on prediction strarting at:  "+tostring(YMDp(0)))

iPred = ind(YMD.eq. YMDp(0)) 
;print(iPred)

olrp = fp->mean_top_net_long_wave_radiation_flux(:, {latS:latN}, :)
u250p = fp->u_component_of_wind(:,8, {latS:latN},:)
u850p = fp->u_component_of_wind(:,14,{latS:latN},:)

olr2d = dim_avg_n_Wrap(olrp, 1)
u2502d = dim_avg_n_Wrap(u250p, 1) 
u8502d = dim_avg_n_Wrap(u850p, 1)
;printVarSummary(olrp)
;printVarSummary(OLR)

do iday = 0, dimsizes(YMDp)-1
   olr2d(iday, :) = olr2d (iday, :) - OLR(iPred, : )
   u2502d(iday, :) = u2502d (iday, :) - U200(iPred, :)
   u8502d(iday, :) = u8502d (iday, :) - U850(iPred, :)
end do

;exit()
;end do




;************************************************
; Normalize by sqrt(avg_var*)
; olr:	 8.65735
; u850:  1.2622
; u200:  3.062939
;************************************************
olr   = olr2d/8.65732         ; (time,lon)
u850  = u8502d/1.2622
u200  = u2502d/3.062939

copy_VarMeta(olr2d, olr)
copy_VarMeta(u8502d, u850)
copy_VarMeta(u2502d, u200)

printVarSummary(olr)
printVarSummary(u850)
printVarSummary(u200)
printMinMax(olr, 0)
printMinMax(u850, 0)
printMinMax(u200, 0)

;exit()


;************************************************
; Combine the normalized data into one variable
;************************************************

ntim = dimsizes(timep)
  cdata     = new ( (/3*mlon,ntim/), typeof(olr), getFillValue(olr))
  do ml=0,mlon-1
     cdata(ml       ,:) = (/  olr(:,ml) /)
     cdata(ml+  mlon,:) = (/ u850(:,ml) /)
     cdata(ml+2*mlon,:) = (/ u200(:,ml) /)
  end do


;*******************
; read in EOF file

  f_eof = addfile("/glade/campaign/univ/uchi0014/benchmark-dev/MJO/eof_era5_1980-2015.nc", "r")
  eof_cdata = f_eof->eof_cdata
  print("==============")
  ;printVarSummary(eof_cdata)
  ;printMinMax(eof_cdata, True)

  eof_ts_cdata = eofunc_ts(cdata,eof_cdata,False)   ; (neof,3*ntim)
  print("==============")
  ;printVarSummary(eof_ts_cdata)
  printMinMax(eof_ts_cdata, True)

;  exit()

;************************************************
; For clarity, explicitly extract each variable. Create time series 
;************************************************

  nvar = 3  ; "olr", "u850", "u200"
  ceof = new( (/nvar,neof,mlon/), typeof(cdata), getFillValue(cdata))

  do n=0,neof-1
     ceof(0,n,:) = eof_cdata(n,0:mlon-1)      ; olr
     ceof(1,n,:) = eof_cdata(n,mlon:2*mlon-1) ; u850
     ceof(2,n,:) = eof_cdata(n,2*mlon:)       ; u200
  end do

  ceof!0   = "var"
  ceof!1   = "eof"
  ceof!2   = "lon"   
  ceof&lon =  olr&lon

  ceof_ts        = new( (/nvar,neof,ntim/), typeof(cdata), getFillValue(cdata))
  ceof_ts(0,:,:) = eofunc_ts_Wrap( olr(lon|:,time|:),ceof(0,:,:),False)   ; (0,neof,ntim)
  ceof_ts(1,:,:) = eofunc_ts_Wrap(u850(lon|:,time|:),ceof(1,:,:),False)   ; (1,neof,ntim)
  ceof_ts(2,:,:) = eofunc_ts_Wrap(u200(lon|:,time|:),ceof(2,:,:),False)   ; (2,neof,ntim)


  ;printMinMax(ceof_ts, True)


;************************************************
; Normalize the multivariate EOF 1&2 component time series
; Compute (PC1^2+PC2^2): values > 1 indicate "strong" periods
;************************************************

  PC1_std =  stddev(eof_ts_cdata(0,:))
  PC2_std =  stddev(eof_ts_cdata(1,:))
  eof_ts_cdata(0,:) = eof_ts_cdata(0,:)/stddev(eof_ts_cdata(0,:))
  eof_ts_cdata(1,:) = eof_ts_cdata(1,:)/stddev(eof_ts_cdata(1,:))

  mjo_ts_index      = eof_ts_cdata(0,:)^2 + eof_ts_cdata(1,:)^2 
  ;mjo_ts_index_smt  = runave(mjo_ts_index, 91, 0) ; 91-day running mean

  nGood   = num(.not.ismissing(mjo_ts_index))     ; # non-missing
  nStrong = num(mjo_ts_index .ge. 1.0)
  print("nGood="+nGood+"   nStrong="+nStrong+"   nOther="+(nGood-nStrong))


;************************************************
; Write PC results to netCDF for use in another example.
;************************************************
  mjo_ts_index!0    = "time"
  mjo_ts_index&time = timep 
  mjo_ts_index@long_name = "MJO PC INDEX" 
  mjo_ts_index@info      = "(PC1^2 + PC2^2)" 

  PC1           = eof_ts_cdata(0,:)
  PC1!0         = "time"
  PC1&time      =  timep
  PC1@long_name = "PC1"
  PC1@info      = "PC1/stddev(PC1)"

  PC2           = eof_ts_cdata(1,:)
  PC2!0         = "time"
  PC2&time      =  timep
  PC2@long_name = "PC2"
  PC2@info      = "PC2/stddev(PC2)"

  diro = "./"
  filo = "MJO_PC_INDEX_PRED_"+tostring(YMDp(0))+"ens_1.nc"
  system("/bin/rm -f "+diro+filo)   ; remove any pre-existing file
  ncdf = addfile(diro+filo,"c")     ; open output netCDF file
                                    ; make time an UNLIMITED dimension 
  filedimdef(ncdf,"time",-1,True)   ; recommended  for most applications
                                    ; output variables directly
  ncdf->MJO_INDEX = mjo_ts_index    
  ncdf->PC1       = PC1     
  ncdf->PC2       = PC2
  ncdf->YMDp  = YMDp
  ncdf->PC1_std   = PC1_std
  ncdf->PC2_std   = PC2_std

  print("~~~~~~~~~~:    "+stddev(eof_ts_cdata(0,:)))
  print("----------:    "+stddev(eof_ts_cdata(1,:)))

  ;print(PC1(::50))

;exit()
end do

end
