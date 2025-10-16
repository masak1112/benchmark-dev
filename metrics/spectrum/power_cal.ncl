


begin

files = systemfunc("ls /glade/campaign/univ/uchi0014/benchmark-dev/predictions/pangu*2019*ens_1.nc")
print(files)
nfiles = dimsizes(files)

power_z500 = new((/nfiles, 46, 180/), float)
power_u250 = new((/nfiles, 46, 180/), float)
power_tp = new((/nfiles, 46, 180/), float)


do i = 0, nfiles - 1,  1

print("working on : "+files(i))
f1 = addfile(files(i), "r")
;printVarSummary(f1->geopotential)
;f2 = addfile(files(i-1), "r")
z500 = f1->geopotential(:, 11, :, :)
u250 = f1->u_component_of_wind(:, 8, :, :)
tp = f1->total_precipitation_24hr(:, :, :)

power_z500(i, :, :)= (/wave_number_spc(z500/9.8, "F")/)
power_u250(i, :, :)= (/wave_number_spc(u250, "F")/)
power_tp(i, :, :)= (/wave_number_spc(tp, "F")/)

end do

unit = wave_number_spc(u250, "F")

;fw = addfile(''"")
fw = addfile("./power_spectrum_s2s.nc","c")
fw->power_z500 = power_z500
fw->power_u250 = power_u250
fw->power_tp  = power_tp

fw->wavenumber = unit&wave_number


end
