 TU DELFT - DEOS
=====================================================
MASTER RESULTFILE: 	/home/banyai/CSOM_ASC_2PS_proc/SLC/20050604/master.res

This file has been created with roipac2doris at Mon Feb 1 14:25:45 CET 2016
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
Volume_ID: 					40943               
Volume_identifier: 				40943
Volume_set_identifier: 				40943
(Check)Number of records in ref. file: 		27181
Product type specifier: 		 	ASAR
Location and date/time of product creation: 	GENERATED AT dummyvariable at dummyvariable
Scene identification: 				dummyvariable
Scene location: 				dummyvariable
Leader file: 					dummyvariable
Scene_centre_latitude: 				46.3430787796
Scene_centre_longitude: 			26.9215447876
Radar_wavelength (m): 				0.0562356424
First_pixel_azimuth_time (UTC): 		04-Jun-2005 19:54:28.8884679624
Pulse_Repetition_Frequency (nominal, Hz): 	1652.41576
Total_azimuth_band_width (Hz):                  1509.85038776043000000000
Xtrack_f_DC_constant (Hz, early edge):          -703.64804438203009
Xtrack_f_DC_linear (Hz/s, early edge):          0
Xtrack_f_DC_quadratic (Hz/s/s, early edge):     0
Range_time_to_first_pixel (2way) (ms): 		5.5174389096
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
Data_output_file: /home/banyai/CSOM_ASC_2PS_proc/INSAR_20050604/20050604_crop.slc
Data_output_format:                             complex_real4
First_line (w.r.t. original_image):  1200
Last_line (w.r.t. original_image):   21000
First_pixel (w.r.t. original_image): 300
Last_pixel (w.r.t. original_image):  3300
*******************************************************************
* End_crop:_NORMAL
*******************************************************************
 
*******************************************************************
*******************************************************************


   *====================================================================*
   |                                                                    |
       Following part is appended at: Mon Feb  1 15:40:04 2016
                 Using Doris version  4.06-beta2 (28-12-2011)
		     build 	Fri Apr 11 04:19:23 2014          
   |                                                                    |
   *--------------------------------------------------------------------*



*******************************************************************
*_Start_precise_orbits:
*******************************************************************
	t(s)	X(m)	Y(m)	Z(m)
NUMBER_OF_DATAPOINTS: 			29
71662.000000	4700976.418	1993441.275	5016266.152
71663.000000	4697010.473	1989666.429	5021463.940
71664.000000	4693038.906	1985890.013	5026656.267
71665.000000	4689061.723	1982112.033	5031843.127
71666.000000	4685078.927	1978332.494	5037024.514
71667.000000	4681090.522	1974551.401	5042200.423
71668.000000	4677096.512	1970768.757	5047370.847
71669.000000	4673096.901	1966984.570	5052535.782
71670.000000	4669091.694	1963198.842	5057695.222
71671.000000	4665080.894	1959411.580	5062849.160
71672.000000	4661064.506	1955622.788	5067997.593
71673.000000	4657042.533	1951832.470	5073140.513
71674.000000	4653014.981	1948040.633	5078277.916
71675.000000	4648981.852	1944247.280	5083409.795
71676.000000	4644943.152	1940452.418	5088536.146
71677.000000	4640898.884	1936656.050	5093656.962
71678.000000	4636849.052	1932858.181	5098772.239
71679.000000	4632793.661	1929058.817	5103881.970
71680.000000	4628732.715	1925257.962	5108986.150
71681.000000	4624666.218	1921455.622	5114084.774
71682.000000	4620594.174	1917651.801	5119177.836
71683.000000	4616516.587	1913846.504	5124265.330
71684.000000	4612433.462	1910039.736	5129347.251
71685.000000	4608344.803	1906231.502	5134423.594
71686.000000	4604250.614	1902421.807	5139494.352
71687.000000	4600150.899	1898610.656	5144559.521
71688.000000	4596045.663	1894798.054	5149619.095
71689.000000	4591934.909	1890984.005	5154673.068
71690.000000	4587818.642	1887168.516	5159721.435

*******************************************************************
* End_precise_orbits:_NORMAL
*******************************************************************


   *====================================================================*
   |                                                                    |
       Following part is appended at: Mon Feb  1 15:51:21 2016
                 Using Doris version  4.06-beta2 (28-12-2011)
		     build 	Fri Apr 11 04:19:23 2014          
   |                                                                    |
   *--------------------------------------------------------------------*



*******************************************************************
*_Start_sim_amplitude:
*******************************************************************
DEM source file:                      	/home/banyai/CSOM_ASC_2PS_proc/DEM/srtm_42_03.bin
Min. of input DEM:                    	206
Max. of input DEM:                    	2191
Data_output_file:                     	master_sam.raw
Data_output_format:                   	real4
First_line (w.r.t. original_master):  	1200
Last_line (w.r.t. original_master):   	21000
First_pixel (w.r.t. original_master): 	300
Last_pixel (w.r.t. original_master):  	3300
Multilookfactor_azimuth_direction:    	1
Multilookfactor_range_direction:      	1
Number of lines (multilooked):        	19801
Number of pixels (multilooked):       	3001
*******************************************************************
* End_sim_amplitude:_NORMAL
*******************************************************************

   Current time: Mon Feb  1 15:52:47 2016


*******************************************************************
*_Start_master_timing:
*******************************************************************
Correlation method 			: 	magfft (4096,2048)
Number of correlation windows used 	: 	30 of 30
Estimated translation master w.r.t. synthetic amplitude (master-dem):
  Positive offsetL: master image is to the bottom
  Positive offsetP: master image is to the right
Coarse_correlation_translation_lines    : 	78
Coarse_correlation_translation_pixels   : 	-2
Master_azimuth_timing_error             : 	-0.0472036 sec.
Master_range_timing_error               : 	5.20625e-08 sec.
*******************************************************************
* End_master_timing:_NORMAL
*******************************************************************

   Current time: Mon Feb  1 16:46:52 2016
