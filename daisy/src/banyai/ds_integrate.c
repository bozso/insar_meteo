//
// ds_integrate
//
// input files
//             dominant.xyd  asc_master.reso dsc_master.reso
//
// Compute the parameters of observation plane, 
// estimate east-west (ev) and up-down (ud) velocities 
// relative to the reference dominant points (DSs).
// 

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define WA  6378137.0            // WGS-84
#define WB  6356752.3142         // WGS-84
#define E2  (WA*WA-WB*WB)/WA/WA
#define distance(x,y,z) sqrt(y*y+x*x+z*z)

typedef struct { double x; double y; double z; // [m,rad]
        	 double f; double l; double h; } station;

void cart_ell ( station *sta );
void ell_cart ( station *sta );

void      axd ( double  a1, double  a2, double a3,
                double  d1, double  d2, double d3,
                double *n1, double *n2, double *n3 );

void closest_appr ( double *poli, int pd, double tf, double tl, station *ps, station *sat);
void    azim_elev ( station ps, station sat, double *azim, double *inc);

void    movements (station ps, double azi1, double inc1, float v1, 
                   double azi2, double inc2, float v2, float *up, float *east, FILE *lo);

void  pause (int i);

// ------------------------------------------------------
 
 int main(int argc, char *argv[])
{
   int i,j,n=0;                                                     
   station ps, sat;
   double azi1, inc1, azi2, inc2;
   double ft1, lt1,               // first and llast time of orbit files
          ft2, lt2,               // first and llast time of orbit files
          *pol1, *pol2;           // orbit polinomials
   int    dop1,dop2;              // degree of orbit polinomials
   
   float la,fi,he,v1,v2, up,east;

   char *buf, 
        *out="integrate.xyi",     // output files 
        *log="integrate.log";     // output files

   FILE *ind,*ino1,*ino2,*ou,*lo;     


  if (( buf = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate BUF");  exit(1); }

//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]);    
//  printf("%s\n",argv[3]);   
     
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
  printf("\n +                       ds_integrate                          +");    
  printf("\n +        compute the east-west and up-down velocities         +");
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc<4)
 {
  printf("\n usage:                                                      \n");
  printf("\n    ds_integrate dominant.xyd asc_master.porb dsc_master.porb\n");
  printf("\n              dominant.xyd  - (1st) dominant DSs data file   ");  
  printf("\n           asc_master.porb  - (2nd) ASC polynomial orbit file");  
  printf("\n           dsc_master.porb  - (3rd) DSC polynomial orbit file\n");      
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
  exit(1);
 }

  if( (ind  = fopen(argv[1],"rt")) == NULL )
  { printf("\n  %s data file not found ! ",argv[1]); exit(1); }
  if( (ino1 = fopen(argv[2],"rt")) == NULL )
  { printf("\n  %s data file not found ! ",argv[2]); exit(1); }
  if( (ino2 = fopen(argv[3],"rt")) == NULL )
  { printf("\n  %s data file not found ! ",argv[3]); exit(1); }
  if( (ou = fopen(out,"w+t")) == NULL )
  { printf("\n  OUT data file not found ! "); exit(1); }
  if( (lo = fopen(log,"w+t")) == NULL )
  { printf("\n  LOG Data file not found ! "); exit(1); }

    printf("\n  inputs:   %s\n          %s\n          %s",argv[1],argv[2],argv[3]);
   printf("\n\n outputs:  %s\n           %s\n",out,log); 

   fprintf(lo,"\n %s %s %s %s\n",argv[0],argv[1],argv[2],argv[3]); 
    fprintf(lo,"\n  inputs:   %s\n          %s\n          %s",argv[1],argv[2],argv[3]);
   fprintf(lo,"\n\n outputs:  %s\n           %s\n",out,log); 
 

// -----------------------------------------------------------   
  fscanf(ino1,"%d %lf %lf",&dop1,&ft1,&lt1); // read orbit
  dop1++;
    if (( pol1 = (double *) malloc(dop1*3 * sizeof(double)) ) == NULL)
  { printf("\nNot enough memory to allocate POL1");  exit(1); }
    for(i=0;i<3;i++)
    for(j=0;j<dop1;j++)
    fscanf(ino1," %lf",(pol1+i*dop1+j));
    fclose(ino1);
// ---------------------------------------
  fscanf(ino2,"%d %lf %lf",&dop2,&ft2,&lt2);  // read orbit
  dop2++;
    if (( pol2 = (double *) malloc(dop2*3 * sizeof(double)) ) == NULL)
  { printf("\nNot enough memory to allocate POL2");  exit(1); }
    for(i=0;i<3;i++)
    for(j=0;j<dop2;j++)
    fscanf(ino2," %lf",(pol2+i*dop2+j));
    fclose(ino2);
// -------------------------------------------------------------

//    details:
//    fprintf(lo,"    longitude       latitude       height     azi1   inc1    v1    azi2    inc2    v2    strike & tilt   tilt  strike & tilt\n");
//    fprintf(lo,"                                                                                             azimuts     angle   movements\n\n"); 
      
    while( fscanf(ind,"%f %f %f %f %f",&la,&fi,&he,&v1,&v2) > 0 )
   {
     ps.f=fi/180.0*M_PI;
     ps.l=la/180.0*M_PI;
     ps.h=he; 
     ell_cart(&ps); 
     
     closest_appr(pol1,dop1,ft1,lt1,&ps,&sat);                  // ******************
        azim_elev(ps, sat, &azi1, &inc1);                       // ******************
                 
     closest_appr(pol2,dop2,ft2,lt2,&ps,&sat);                  // ******************
        azim_elev(ps, sat, &azi2, &inc2);                       // ******************

     movements(ps,azi1,inc1,v1, azi2,inc2,v2, &up, &east, lo);  // ******************
    
     fprintf(ou,"%16.7e %15.7e %9.3f %7.3f %7.3f\n", la,fi,he,east,up);  
     n++;  if((n%1000)==0) printf("\n %6d ...",n);                                 
   }

      printf("\n %6d",n); 

      printf("\n\n Records of %s file:\n",out); 
      printf("\n longitude latitude  height  ew_v   up_v"); 
      printf("\n (     degree          m       mm/year )");  


     fprintf(lo,"\n DSs  %6d\n",n); 
     fprintf(lo,"\n Records of %s file:\n",out); 
     fprintf(lo,"\n longitude latitude  height  ew_v   up_v"); 
     fprintf(lo,"\n (     degree          m       mm/year )\n\n");  
 
printf("\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
printf("\n +                     end    ds_integrate                     +");
printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(1);
      
}  // end main   
// ***********************************************************

// -----------------------------------------------------------
  void movements(station ps, double azi1, double inc1, float v1, 
                             double azi2, double inc2, float v2, float *up, float *east, FILE *lo)
{
   double  a1, a2, a3;       // unit vector of sat1
   double  d1, d2, d3;       // unit vector of sat2 
   double  n1, n2, n3,  ln;  // 3D vector and its legths
   double  s1, s2, s3,  ls;  // 3D vector and its legths   
   double zap, zdp, zad;     // angles in observation plain
   double az, ti, hl;
   double al1,al2,in1,in2;
   double sm, vm;            // movements in observation plain

   al1=azi1/180.0*M_PI;
   in1=inc1/180.0*M_PI;
   al2=azi2/180.0*M_PI;
   in2=inc2/180.0*M_PI;
//-------------------------            
   a1=-sin(al1)*sin(in1);    // E
   a2=-cos(al1)*sin(in1);    // N
   a3=          cos(in1);    // U         
   d1=-sin(al2)*sin(in2);
   d2=-cos(al2)*sin(in2);
   d3=          cos(in2);     
//-------------------------------------------------------
   axd(a1,a2,a3, d1,d2,d3, &n1,&n2,&n3); // normal vector   
   ln=sqrt(n1*n1+n2*n2+n3*n3);
   zad=asin(ln);                         // agle between two unit vector
//-------------------------------------------------------    
   n1=n1/ln;
   n2=n2/ln; 
   n3=n3/ln;       
   az=atan(n1/n2);
     
   hl=sqrt(n1*n1+n2*n2);
   ti=atan(n3/hl);
  
   s1=-n3*sin(az);
   s2=-n3*cos(az);
   s3= hl;

   n1= s1;    //  vector in the plain
   n2= s2;
   n3= s3;
//---------------------------------------
   axd(a1,a2,a3, n1,n2,n3, &s1,&s2,&s3);
   ls=sqrt(s1*s1+s2*s2+s3*s3);
   zap=asin(ls);   // alfa 

   axd(d1,d2,d3, n1,n2,n3, &s1,&s2,&s3);
   ls=sqrt(s1*s1+s2*s2+s3*s3);
   zdp=asin(ls);     // beta
   
    sm=(v2/cos(zdp) - v1/cos(zap))/(tan(zap)+tan(zdp));   // strike movement
    vm= v1/cos(zap)+tan(zap)*sm;                          // tilt movement 
     
    *up  = vm/cos(ti);         // biased Up   component
    *east= sm/cos(az);         // biased East component

//  details:    
//  fprintf(lo,"%16.7e %15.7e %9.3f %7.2lf %6.2lf %6.2f %7.2lf %6.2f %6.2f %7.2lf %7.2lf %6.2lf %6.2lf %6.2lf\n",
//             ps.l/M_PI*180.0,ps.f/M_PI*180.0,ps.h,
//             azi1,inc1,v1, azi2,inc2,v2,
//             90.0+az/M_PI*180.0,
//             180.0+az/M_PI*180.0,
//             ti/M_PI*180.0,
//             sm,vm);  
   
} // end  movement
//++++++++++++++++++++++++++++++++++++++++++

//-----------------------------------------
 void axd( double  a1, double  a2, double  a3,
           double  d1, double  d2, double  d3,
           double *n1, double *n2, double *n3 )
{
// vectorial multiplication a x d
//  
   *n1=a2*d3-a3*d2;
   *n2=a3*d1-a1*d3;
   *n3=a1*d2-a2*d1;
} // end axd
// ---------------------------------------------------------------
 void azim_elev(station ps, station sat, double *azi, double *inc)
{ 
//
// topocentric parameters in PS local system
//
     double xf,yf,zf, xl,yl,zl, t0;

     xf = sat.x - ps.x; // cart system
     yf = sat.y - ps.y;     
     zf = sat.z - ps.z; 

     xl= - sin(ps.f)*cos(ps.l)*xf - sin(ps.f)*sin(ps.l)*yf + cos(ps.f)*zf ;
     yl= -           sin(ps.l)*xf +           cos(ps.l)*yf                 ;
     zl= + cos(ps.f)*cos(ps.l)*xf + cos(ps.f)*sin(ps.l)*yf + sin(ps.f)*zf ;

     t0=distance(xl,yl,zl);

     *inc = acos(zl/t0)/M_PI*180.0;

     if(xl==0.0) xl=0.000000001;   
     *azi=atan(fabs(yl/xl));   
     if((xl<0.0) && (yl>0.0)) *azi=M_PI-*azi;
     if((xl<0.0) && (yl<0.0)) *azi=M_PI+*azi;
     if((xl>0.0) && (yl<0.0)) *azi=2.0*M_PI-*azi;   
     *azi=*azi/M_PI*180.0;                            //   azimut ps->sat 
     if(*azi > 180.0) *azi -=180.0; else *azi +=180.0; //  azimut sat->ps 

//printf("\n azi  inc  %12.4lf %12.4lf",*azi,*inc);     
//pause(-3);     
}

                               
// -----------------------------------------------------------

  void closest_appr(double *poli,int pd,double tfp,double tlp,
                                     station *ps, station *sat)
{
//
// compute the sat position using closest approache
//                                        
  double    tf,   tl,   tm; // first, last and middle time
  double    vs,   ve,   vm; // vectorial products
  double vxs,vys,vzs;       // sat velocities
  double lvs,lps;           // vector length

  double dx,dy,dz;
 
  int i,itr;

  tf=tfp-tfp;
  tl=tlp-tfp; 

// first S1 position 

   sat->x=sat->y=sat->z = vxs=vys=vzs = 0.0;

   for(i=0;i<pd;i++) sat->x += *(poli     +i) * pow(tf,1.0*i); 
   for(i=0;i<pd;i++) sat->y += *(poli+  pd+i) * pow(tf,1.0*i);    
   for(i=0;i<pd;i++) sat->z += *(poli+2*pd+i) * pow(tf,1.0*i); 
   dx=sat->x - ps->x;
   dy=sat->y - ps->y;  
   dz=sat->z - ps->z; 
   lps=distance(dx,dy,dz); 

   for(i=1;i<pd;i++) vxs += i * *(poli     +i) * pow(tf,1.0*(i-1)); 
   for(i=1;i<pd;i++) vys += i * *(poli+  pd+i) * pow(tf,1.0*(i-1));    
   for(i=1;i<pd;i++) vzs += i * *(poli+2*pd+i) * pow(tf,1.0*(i-1)); 
   lvs=distance(vxs,vys,vzs);

   vs = vxs/lvs*(dx)/lps + vys/lvs*(dy)/lps + vzs/lvs*(dz)/lps; 

// last S1 position 

   sat->x=sat->y=sat->z = vxs=vys=vzs = 0.0;   

   for(i=0;i<pd;i++) sat->x += *(poli     +i) * pow(tl,1.0*i); 
   for(i=0;i<pd;i++) sat->y += *(poli+  pd+i) * pow(tl,1.0*i);    
   for(i=0;i<pd;i++) sat->z += *(poli+2*pd+i) * pow(tl,1.0*i); 
   dx=sat->x - ps->x;
   dy=sat->y - ps->y;  
   dz=sat->z - ps->z; 
   lps=distance(dx,dy,dz); 

   for(i=1;i<pd;i++) vxs += i * *(poli     +i) * pow(tl,1.0*(i-1)); 
   for(i=1;i<pd;i++) vys += i * *(poli+  pd+i) * pow(tl,1.0*(i-1));    
   for(i=1;i<pd;i++) vzs += i * *(poli+2*pd+i) * pow(tl,1.0*(i-1)); 
   lvs=distance(vxs,vys,vzs);

   ve = vxs/lvs*(dx)/lps + vys/lvs*(dy)/lps + vzs/lvs*(dz)/lps; 

        itr=0;
   do {
        tm =(tf+tl)/2.0;

//     middle S1 position 

        sat->x=sat->y=sat->z = vxs=vys=vzs = 0.0;

        for(i=0;i<pd;i++) sat->x += *(poli     +i) * pow(tm,1.0*i); 
        for(i=0;i<pd;i++) sat->y += *(poli+  pd+i) * pow(tm,1.0*i);    
        for(i=0;i<pd;i++) sat->z += *(poli+2*pd+i) * pow(tm,1.0*i); 
        dx=sat->x - ps->x;
        dy=sat->y - ps->y;  
        dz=sat->z - ps->z; 
        lps=distance(dx,dy,dz); 

        for(i=1;i<pd;i++) vxs += i * *(poli     +i) * pow(tm,1.0*(i-1)); 
        for(i=1;i<pd;i++) vys += i * *(poli+  pd+i) * pow(tm,1.0*(i-1));    
        for(i=1;i<pd;i++) vzs += i * *(poli+2*pd+i) * pow(tm,1.0*(i-1)); 
        lvs=distance(vxs,vys,vzs);

        vm = vxs/lvs*(dx)/lps + vys/lvs*(dy)/lps + vzs/lvs*(dz)/lps; 

        if( (vs*vm)>0.0 ) { tf=tm; vs=vm; }   // cahnge start for middle 
        else              { tl=tm; ve=vm; }   // change  end  for middle
 
        itr++;

      } while( fabs(vm) > 1.0e-11 );

} // end closest_appr
          
// ---------------------------------------------------------------------
  void pause(int i)
{
 printf("\n\n Pause(%1d) => Ctrl c to exit - Enter to continue !\n",i);
 getchar();
}
//*****************************

 void cart_ell( station *sta )
{
// from cartesian to ellipsoidal
// coordinates

  double n, p,o,so,co, x,y,z;

 n=(WA*WA-WB*WB);
 x=sta->x; y=sta->y; z=sta->z;
 p=sqrt(x*x+y*y);

 o=atan(WA/p/WB*z);
 so=sin(o); co=cos(o);
 o=atan((z+n/WB*so*so*so)/(p-n/WA*co*co*co));
 so=sin(o); co=cos(o);
 n=WA*WA/sqrt(WA*co*co*WA+WB*so*so*WB);

 sta->f = o;
 o=atan(y/x); if(x<0.0) o+=M_PI;
 sta->l = o;
 sta->h = p/co-n;
 if(x==y && y==z && z==0.0) // sta->f=sta->l=sta->h=0.0;
  {sta->f=M_PI/4.0;
   sta->l=M_PI/10.0;
   sta->h=0.0;
     n=WA/sqrt(1.0-E2*sin(sta->f)*sin(sta->f));
   sta->x=(         n+sta->h)*cos(sta->f)*cos(sta->l);
   sta->y=(         n+sta->h)*cos(sta->f)*sin(sta->l);
   sta->z=((1.0-E2)*n+sta->h)*sin(sta->f);
  }
}  // end of cart_ell

// *************************

 void ell_cart( station *sta )
{
// from ellipsoidal to cartesian
// coordinates
   double fi,la,n;
   fi=sta->f;
   la=sta->l;
   n=WA/sqrt(1.0-E2*sin(fi)*sin(fi));
   sta->x=(         n+sta->h)*cos(fi)*cos(la);
   sta->y=(         n+sta->h)*cos(fi)*sin(la);
   sta->z=((1.0-E2)*n+sta->h)*sin(fi);  
}  // end of ell_cart

// ************************* 
