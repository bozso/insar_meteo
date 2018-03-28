 TU DELFT - DEOS
=====================================================
MASTER RESULTFILE: 	/home/banyai/CSOM_DES_18PS_proc/SLC/20051215/master.res

This file has been created with roipac2doris at Tue Feb 9 13:44:41 CET 2016
InSAR PROCESSOR: 	Doris
Version of software: 	Version ??
Compiled at: 		??
Creation of this file: 	??
=====================================================


Start_process_control
readfiles:		1
precise_orbits:		1
crop:			1
sim_amplitude:		1
master_timing:		1
oversample:		0
resample:		0
filt_azi:		0
filt_range:		0
End_process_control

*******************************************************************
*_Start_readfiles:
*******************************************************************
Volume file: 					dummyvariable
Volume_ID: 					46403               
Volume_identifier: 				46403
Volume_set_identifier: 				46403
(Check)Number of records in ref. file: 		27181
Product type specifier: 		 	ASAR
Location and date/time of product creation: 	GENERATED AT dummyvariable at dummyvariable
Scene identification: 				dummyvariable
Scene location: 				dummyvariable
Leader file: 					dummyvariable
Scene_centre_latitude: 				46.3532970128
Scene_centre_longitude: 			25.0970847603
Radar_wavelength (m): 				0.0562356424
First_pixel_azimuth_time (UTC): 		15-Dec-2005 08:37:32.6896957879
Pulse_Repetition_Frequency (nominal, Hz): 	1652.41576
Total_azimuth_band_width (Hz):                  1509.83399738498200000000
Xtrack_f_DC_constant (Hz, early edge):          214.29430604586915
Xtrack_f_DC_linear (Hz/s, early edge):          0
Xtrack_f_DC_quadratic (Hz/s/s, early edge):     0
Range_time_to_first_pixel (2way) (ms): 		5.5201461625
Range_sampling_rate (computed, MHz): 		19.2076800000
Total_range_band_width (MHz):                   16.00000001890318060000
Datafile: 					ROIPACSLC     
Number_of_lines_original: 			27180
Number_of_pixels_original: 			5681
*******************************************************************
* End_readfiles:_NORMAL
*******************************************************************


*******************************************************************
*_Start_crop:                                   master
*******************************************************************
Data_output_file: /home/banyai/CSOM_DES_18PS_proc/INSAR_20051215/20051215_crop.slc
Data_output_format:                             complex_real4
First_line (w.r.t. original_image):  4800
Last_line (w.r.t. original_image):   26000
First_pixel (w.r.t. original_image): 200
Last_pixel (w.r.t. original_image):  3200
*******************************************************************
* End_crop:_NORMAL
*******************************************************************
 
*******************************************************************
*******************************************************************


   *====================================================================*
   |                                                                    |
       Following part is appended at: Tue Feb  9 15:58:06 2016
                 Using Doris version  4.06-beta2 (28-12-2011)
		     build 	Fri Apr 11 04:19:23 2014          
   |                                                                    |
   *--------------------------------------------------------------------*



*******************************************************************
*_Start_precise_orbits:
*******************************************************************
	t(s)	X(m)	Y(m)	Z(m)
NUMBER_OF_DATAPOINTS: 			29
31046.000000	4298687.261	2452984.129	5170757.428
31047.000000	4304237.266	2453889.111	5165721.163
31048.000000	4309782.756	2454790.635	5160679.280
31049.000000	4315323.726	2455688.701	5155631.785
31050.000000	4320860.169	2456583.308	5150578.682
31051.000000	4326392.078	2457474.456	5145519.978
31052.000000	4331919.446	2458362.145	5140455.679
31053.000000	4337442.268	2459246.375	5135385.788
31054.000000	4342960.537	2460127.145	5130310.313
31055.000000	4348474.246	2461004.454	5125229.258
31056.000000	4353983.389	2461878.304	5120142.629
31057.000000	4359487.959	2462748.692	5115050.432
31058.000000	4364987.950	2463615.620	5109952.672
31059.000000	4370483.356	2464479.086	5104849.354
31060.000000	4375974.170	2465339.091	5099740.484
31061.000000	4381460.385	2466195.635	5094626.068
31062.000000	4386941.995	2467048.716	5089506.112
31063.000000	4392418.994	2467898.335	5084380.620
31064.000000	4397891.375	2468744.491	5079249.598
31065.000000	4403359.132	2469587.185	5074113.052
31066.000000	4408822.259	2470426.415	5068970.988
31067.000000	4414280.748	2471262.182	5063823.410
31068.000000	4419734.594	2472094.486	5058670.326
31069.000000	4425183.790	2472923.326	5053511.739
31070.000000	4430628.330	2473748.703	5048347.656
31071.000000	4436068.208	2474570.615	5043178.083
31072.000000	4441503.416	2475389.062	5038003.024
31073.000000	4446933.949	2476204.045	5032822.487
31074.000000	4452359.800	2477015.564	5027636.475

*******************************************************************
* End_precise_orbits:_NORMAL
*******************************************************************


   *====================================================================*
   |                                                                    |
       Following part is appended at: Tue Feb  9 16:07:14 2016
                 Using Doris version  4.06-beta2 (28-12-2011)
		     build 	Fri Apr 11 04:19:23 2014          
   |                                                                    |
   *--------------------------------------------------------------------*



*******************************************************************
*_Start_sim_amplitude:
*******************************************************************
DEM source file:                      	/home/banyai/CSOM_DES_18PS_proc/DEM/srtm_42_03.bin
Min. of input DEM:                    	210
Max. of input DEM:                    	2094
Data_output_file:                     	master_sam.raw
Data_output_format:                   	real4
First_line (w.r.t. original_master):  	4800
Last_line (w.r.t. original_master):   	26000
First_pixel (w.r.t. original_master): 	200
Last_pixel (w.r.t. original_master):  	3200
Multilookfactor_azimuth_direction:    	1
Multilookfactor_range_direction:      	1
Number of lines (multilooked):        	21201
Number of pixels (multilooked):       	3001
*******************************************************************
* End_sim_amplitude:_NORMAL
*******************************************************************

   Current time: Tue Feb  9 16:08:49 2016


*******************************************************************
*_Start_master_timing:
*******************************************************************
Correlation method 			: 	magfft (4096,2048)
Number of correlation windows used 	: 	30 of 30
Estimated translation master w.r.t. synthetic amplitude (master-dem):
  Positive offsetL: master image is to the bottom
  Positive offsetP: master image is to the right
Coarse_correlation_translation_lines    : 	-26
Coarse_correlation_translation_pixels   : 	-5
Master_azimuth_timing_error             : 	0.0157345 sec.
Master_range_timing_error               : 	1.30156e-07 sec.
*******************************************************************
* End_master_timing:_NORMAL
*******************************************************************

   Current time: Tue Feb  9 17:03:31 2016
