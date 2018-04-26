#include <stdio.h>
#include <tgmath.h>
#include <stdlib.h>
#include <string.h>

/* minimum number of arguments:
 *     - argv[0] is the executable name
 *     - argv[1] is the module name */
#define Minarg 2

#define error(string) fprintf(stderr, string)
#define errorln(format, ...) fprintf(stderr, format"\n", __VA_ARGS__)

#define println(format, ...) printf(format"\n", __VA_ARGS__)

#define Log printf("%s\t%d\n", __FILE__, __LINE__)

#define Str_IsEqual(string1, string2) (strcmp((string1), (string2)) == 0)
#define Str_Select(string) Str_IsEqual(argv[1], string)

#define R 6372000      // radius of Earth
#define C 57.295779513 // 180/pi

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define WA  6378137.0            // WGS-84
#define WB  6356752.3142         // WGS-84
#define E2  (WA*WA-WB*WB)/WA/WA
#define distance(x,y,z) sqrt(y*y+x*x+z*z)

typedef struct { float la; float fi;  } psxy;
typedef struct {  int ni; float la; float fi; float he; float ve; } psxys;
typedef struct { double x; double y; double z;                    
	      	 double f; double l; double h; } station;      // [m,rad]

typedef struct { double t; double x; double y; double z; } torb; 

/************************
 * Auxilliary functions *
 ************************/

static void cart_ell( station *sta )
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
 
}  // end of cart_ell

// --------------------------

static void ell_cart( station *sta )
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

static int selectp(float dam, FILE *in1, psxy *in2, int ni, FILE *ou1)
 { 
// The PS"la1,fi1" is selected if the first PS"la2,fi1"
// is closer than the sepration distance "dam" 
// The "dam", "la1,fi1" and "la2,fi1" are interpreted
// on spherical Earth with radius 6372000 m 
//
  int n=0, // no of selected PSs
      m=0, // no of data in in1
      ef;
  float fi1,la1,v1,he1,dhe1, fi2,la2,v2, da;

  float dm=dam/R*C*dam/R*C; // faster run
  
   while( fscanf(in1,"%e %e %e %e %e",&la1,&fi1,&v1,&he1,&dhe1) > 0)
  { 
    ef=0;                 
    do {         
          la2= (in2+ef)->la;
          fi2= (in2+ef)->fi; 

//        da= R * acos( sin(fi1/C)*sin(fi2/C)+cos(fi1/C)*cos(fi2/C)*cos((la1-la2)/C) ); 
//        da= da-dam;

          da= (fi1-fi2)*(fi1-fi2)+(la1-la2)*(la1-la2); // faster run
          da= da-dm;                                   // rewer  PSs

          ef++;  
       } 
    while( (da > 0.0) && ( (ef-1) < ni) ); 

    if( (ef-1) < ni ) {
                         fprintf(ou1,"%16.7e %16.7e %16.7e %16.7e %16.7e\n",la1,fi1,v1,he1,dhe1);
                        n++;
                      }
     m++;  if((m%10000)==0) printf("\n %6d ...",m);               
   }  
   return(n);
 } // end selectp

static void estim_dominant(psxys *buffer, int ps1,int ps2, FILE *lo, FILE *ou)
{
   int i;
   
   double dist, dx,dy,dz, sumw, sumwve;
   station ps,psd;

// coordinates of dominant point - weighted mean

   psd.x = psd.y = psd.z = 0.0;
  
   for(i=0;i<(ps1+ps2);i++)
  { 

//   details:
//   fprintf(lo,"%d %16.7e %15.7e %9.3f %8.3f\n",(buffer+i)->ni,(buffer+i)->la,(buffer+i)->fi,(buffer+i)->he,(buffer+i)->ve );

   ps.f = (buffer+i)->fi/180.0*M_PI;
   ps.l = (buffer+i)->la/180.0*M_PI;
   ps.h = (buffer+i)->he;
   ell_cart( &ps );                    // compute ps.x ps.y ps.z 
   
   if(i<ps1) { psd.x += ps.x/ps1;
               psd.y += ps.y/ps1;
               psd.z += ps.z/ps1; }
   else      { psd.x += ps.x/ps2;
               psd.y += ps.y/ps2;
               psd.z += ps.z/ps2; }     // sum (1/ps1 + 1/ps2) = 2           
  } //end for
  
   psd.x /= 2.0;
   psd.y /= 2.0;                        // weighted meam
   psd.z /= 2.0;  
                 
   cart_ell( &psd ); 

//   details:
//   fprintf(lo,"0 %16.7le %15.7le %9.3lf",psd.l/M_PI*180.0, psd.f/M_PI*180.0, psd.h);    
   
   fprintf(ou,  "%16.7le %15.7le %9.3lf",psd.l/M_PI*180.0, psd.f/M_PI*180.0, psd.h);  
 
// interpolation of ascending velocities
 
   sumwve=sumw=0.0;

   for(i=0;i<ps1;i++)
  {     
    ps.f = (buffer+i)->fi/180.0*M_PI;
    ps.l = (buffer+i)->la/180.0*M_PI;
    ps.h = (buffer+i)->he;
    ell_cart( &ps ); 

    dx= psd.x-ps.x;
    dy= psd.y-ps.y;
    dz= psd.z-ps.z;               
    dist=distance(dx,dy,dz); 

    sumw   +=1.0/dist/dist;              // weight
    sumwve +=(buffer+i)->ve/dist/dist;
  } 
    fprintf(ou," %8.3lf",sumwve/sumw); 

//    details:
//    fprintf(lo," %8.3lf",sumwve/sumw); 

// interpolation of descending velocities
  
   sumwve=sumw=0.0;

   for(i=ps1;i<(ps1+ps2);i++)
  {     
    ps.f = (buffer+i)->fi/180.0*M_PI;
    ps.l = (buffer+i)->la/180.0*M_PI;
    ps.h = (buffer+i)->he;
    ell_cart( &ps );
    dx=   psd.x-ps.x;
    dy=   psd.y-ps.y;
    dz=   psd.z-ps.z;               
    dist= distance(dx,dy,dz); 

    sumw   +=1.0/dist/dist;              // weight
    sumwve +=(buffer+i)->ve/dist/dist;
  }  
    fprintf(ou," %8.3lf\n",sumwve/sumw);

//    details:
//    fprintf(lo," %8.3lf\n",sumwve/sumw);
     
} //end estim_dominant

// -----------------------------------------------------------
static int  cluster(psxys *indata1, int n1, psxys *indata2, int n2, 
               psxys *buffer,  int *nb, float dam)
{ 
  int i,j,k;
  double fv, dla,dfi,dd, la,fi;
  double dm=dam/R*C*dam/R*C; 

     k=j=0;     
     while( ((indata1+k)->ni == 0) && (k < n1) )  k++;   // skip selected PSs
     
     la = (indata1+k)->la; 
     fi = (indata1+k)->fi; 
          
   for(i=k;i<n1;i++)  // 1 for
  {                      
     dla=(indata1+i)->la - la;
     dfi=(indata1+i)->fi - fi;  
     dd = dla*dla + dfi*dfi;                         
      if( ((indata1+i)->ni > 0) && (dd<dm) )
     {
       *(buffer+j)=*(indata1+i); 
       (indata1+i)->ni = 0;
       j++; if(j == *nb) { (*nb)++; buffer = (psxys *) realloc(buffer,*nb * sizeof(psxys)); }
     } // end if       
  } // end  1 for                  
      
   for(i=0;i<n2;i++)  // 2 for
  {                      
     dla=(indata2+i)->la - la;
     dfi=(indata2+i)->fi - fi;  
     dd = dla*dla + dfi*dfi; 
                        
      if( ((indata2+i)->ni >0) && (dd<dm) )
     {                  
       *(buffer+j)=*(indata2+i);
      (indata2+i)->ni = 0;
       j++; if(j == *nb) { (*nb)++; buffer = (psxys *) realloc(buffer,*nb * sizeof(psxys)); }      
     } // end if                               
  } // end  2 for                  

  return(j); 
}  // end cluster  

static void axd( double  a1, double  a2, double  a3,
           double  d1, double  d2, double  d3,
           double *n1, double *n2, double *n3 )
{
// vectorial multiplication a x d
//  
   *n1=a2*d3-a3*d2;
   *n2=a3*d1-a1*d3;
   *n3=a1*d2-a2*d1;
} // end axd

static void movements(station ps, double azi1, double inc1, float v1, 
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

static void azim_elev(station ps, station sat, double *azi, double *inc)
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

static void closest_appr(double *poli,int pd,double tfp,double tlp,
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

static int plc(int i, int j, int n)
{
// position of i-th, j-th element  0
// in the lower triangle           1 2
// stored in vector                3 4 5 

 int k,l;
 k=(i<j) ? i+1 : j+1;
 l=(i>j) ? i+1 : j+1;
 return( (l-1)*l/2+k-1 );
} // end plc

static int chole(double *q, int n)
{
// Cholesky decomposition of
// symmetric positive definit
// normal matrix

 int i,j,k,l,ia,l1;
 double a,sqr,sum;

 for(i=0;i<n;i++) {
  a=*(q+plc(i,i,n));

  if( a<=0.0 ) return(1);
  sqr=sqrt(a);
  for(k=i;k<n;k++) *(q+plc(i,k,n))/=sqr;
  l=i+1; if(l==n) goto end1;
  for(j=l;j<n;j++) for(k=j;k<n;k++) {

   *(q+plc(j,k,n))-= *(q+plc(i,j,n)) * *(q+plc(i,k,n)); }
 }
 end1:
 for(i=0;i<n;i++) {

  ia=plc(i,i,n); *(q+ia)=1.0/ *(q+ia);
  l1=i+1; if(l1==n) goto end2;
  for(j=l1;j<n;j++) {

   sum=0.0;
   for(k=i;k<j;k++) sum+= *(q+plc(i,k,n)) * *(q+plc(k,j,n));
   *(q+plc(i,j,n))=-sum/ *(q+plc(j,j,n)); }
 }
 end2:
 for(i=0;i<n;i++) for(k=i;k<n;k++) { sum=0.0;

  for(j=k;j<n;j++) {
   sum+= *(q+plc(i,j,n)) * *(q+plc(k,j,n)); }
  *(q+plc(i,k,n))=sum; }

 return(0);
 } // end chole

static void ATA_ATL( int m,int u, double *A, double *L, double *ATA, double *ATL)
{
// m - number of measurements
// u - number of unknowns

  int i,j,k,l;
  double *buf;

  if (( buf = (double *) malloc( m * sizeof(double)) ) == NULL)
  { error("\n Not enough memory to allocate BUF \n"); exit(1); }

  for(i=0;i<u;i++) // row of ATPA
  {

// buffer +++++++++++++++++++++++++++++++++
    for(k=0;k<m;k++)
   {  *(buf+k)=0.0;
    for(l=0;l<m;l++) *(buf+k)= *(buf+k) + *(A + l*u + i );
   } 
// buffer +++++++++++++++++++++++++++++++++

   for(l=0;l<m;l++)
   {
    *(ATL+i) = *(ATL+i) + *(buf+l) * *(L+l);
   }

  for(j=i;j<u;j++) // column of ATPA
  {
   k=plc(i,j,u);

   for(l=0;l<m;l++)
   {
    *(ATA+k) = *(ATA+k) + *(buf+l) * *(A+l*u+j);
   }
  }  // end - column of ATPA
  }  // end - row of ATPA
  free(buf);

} // end ATA_ATL

static int  poly_fit ( int m, int u, torb *orb, double *X, char c, FILE *lo )
{
//
// o(t) = a0 + a1*t + a2*t^2 + a3*t^3  + ... 
//    
  int i,j;         
  double t, *A, *L, *ATA, *ATL, mu0=0.0;
   
   if ((   A = (double *) malloc(u * sizeof(double)) ) == NULL)
  { error("\nNot enough memory to allocate A\n"); exit(1); } 
   if (( ATA = (double *) calloc(u*(u+1)/2, sizeof(double)) ) == NULL)
  { error("\nNot enough memory to allocate ATA\n"); exit(1); } 
   if (( ATL = (double *) calloc(u, sizeof(double)) ) == NULL)
  { error("\nNot enough memory to allocate ATL\n"); exit(1); } 
   if ((   L = (double *) calloc( 1, sizeof(double)) ) == NULL)
  { error("\nNot enough memory to allocate L\n"); exit(1); } 

  for(i=0;i<m;i++)
 { 
          if(c=='x') *L= (orb+i)->x;
     else if(c=='y') *L= (orb+i)->y; 
     else if(c=='z') *L= (orb+i)->z;           
     t= (orb+i)->t - orb->t;   
     for(j=0;j<u;j++) *(A+j)= pow(t,1.0*j);     
     ATA_ATL( 1, u, A, L, ATA, ATL);    
 } 
   if( chole( ATA,u ) != 0 )
   { error("\n Error - singular normal matrix ! \n"); exit(0); } 

     for(i=0;i<u;i++) // update of unknowns
     {                 *(X+i) = 0.0;
     for(j=0;j<u;j++) *(X+i) = *(X+i) + *(ATA+plc(i,j,u)) * *(ATL+j); }
    
  for(i=0;i<m;i++)
 { 
          if(c=='x') *L= -(orb+i)->x;
     else if(c=='y') *L= -(orb+i)->y; 
     else if(c=='z') *L= -(orb+i)->z; 
     t= (orb+i)->t - orb->t;
     
     for(j=0;j<u;j++) *L += *(X+j)* pow(t,1.0*j); 
     mu0 += *L * *L;
 }
   mu0=sqrt(mu0/(m-u)*1.0);

   fprintf(lo,"  mu0= %8.4lf dof= %d",mu0,m-u);
//   fprintf(lo,"\n\n         coefficients                  std\n");
  
   printf("\n\n mu0= %8.4lf",mu0);     
   printf("     dof= %d",m-u);
   printf("\n\n         coefficients                  std\n");

   for(j=0;j<u;j++) 
  { 
        printf("\n%2d %23.15e   %23.15e",j, *(X+j), mu0*sqrt( *(ATA+plc(j,j,u)) )); 
 
//  fprintf(lo,"\n%2d %23.15e   %23.15e",j, *(X+j), mu0*sqrt( *(ATA+plc(j,j,u)) ));
  
  } 
     
 return(1); 
} // end poly_fit

// -------------------------------------------------

static void change_ext ( char *name, char *ext )
{
// change the extent of name for ext

  int  i=0;
  while( *(name+i) != '.' && *(name+i) != '\0') i++;
  *(name+i)='\0';

  sprintf(name,"%s.%s",name,ext);

 } // end change_ext


/****************
 * Main modules *
 ****************/

 int data_select(int argc, char *argv [])
{
   int i,n,ni1,ni2;
   psxy *indata;
   
   char *inp1;                      // ASC input file
   char *inp2;                      // DSC input file     
   char *out1;                      // output file
   char *out2;                      // output file  

   char *logf="data_select.log";    // log output file
 
   FILE *in1,*in2,*ou1, *ou2, *log;    

   float dam;
   float la,fi,v,he,dhe;

  if (( out1 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\n Not enough memory to allocate OUT1\n");  exit(1); }
  if (( out2 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\n Not enough memory to allocate OUT2\n");  exit(1); }

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                  ps_data_select                    +\
            \n + adjacent ascending and descending PSs are selected +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    
    if(argc - Minarg<3)
   {
    printf("\n   usage:  ps_data_select asc_data.xy dsc_data.xy 100  \n\
            \n           asc_data.xy  - (1st) ascending  data file\
            \n           dsc_data.xy  - (2nd) descending data file\
            \n           100          - (3rd) PSs separation (m)\n\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
    exit(1);
   }

    if( (log = fopen(logf,"w+t")) == NULL )
   { printf("\n  LOG file not found ! "); exit(1); }

    fprintf(log,"\n %s %s %s %s\n",argv[0],argv[1],argv[2],argv[3]);


    sprintf(out1,"%s%s",argv[1],"s");
    sprintf(out2,"%s%s",argv[2],"s");   
         printf("\n  input: %s\n output: %s\n",argv[1],out1);
         printf("\n  input: %s\n output: %s\n",argv[2],out2); 

    fprintf(log,"\n  input: %s\n output: %s\n",argv[1],out1);
    fprintf(log,"\n  input: %s\n output: %s\n",argv[2],out2); 
      
  if( (in1 = fopen(argv[1],"rt")) == NULL )
  { error("\n  ASC Data file not found !\n"); exit(1); }
  if( (in2 = fopen(argv[2],"rt")) == NULL )
  { error("\n  DSC Data file not found !\n"); exit(1); }

  if( (ou1 = fopen(out1,"w+t")) == NULL )
  { error("\n  OU1 Data file not found !\n"); exit(1); }
  if( (ou2 = fopen(out2,"w+t")) == NULL )
  { error("\n  OU2 Data file not found !\n"); exit(1); }

//---------------------------------------------------------------
   sscanf(argv[3],"%f",&dam);

        printf("\n Appr. PSs separation %5.1f (m)\n",dam); 
   fprintf(log,"\n Appr. PSs separation %5.1f (m)",  dam); 
//----------------------------------------------------------------

   ni1=0;
   while(fscanf(in1,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe)>0) ni1++;
   rewind(in1); 
 
   ni2=0;
   while(fscanf(in2,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe)>0) ni2++;
   rewind(in2); 

//  Copy data to memory 
    
   if (( indata = (psxy *) malloc(ni2 * sizeof(psxy)) ) == NULL)
   { printf("\nNot enough memory to allocate indata 1");  exit(1); }

   for(i=0;i<ni2;i++)
   {
     fscanf(in2,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe);
     (indata+i)->la=la;
     (indata+i)->fi=fi;               
   }

//-------------------------------------------------------------------  

        printf("\n\n %s  PSs %d\n",argv[1],ni1);
   fprintf(log,"\n\n %s  PSs %d",  argv[1],ni1);
 
   printf("\n Select PSs ...\n");  
   n=selectp(dam, in1, indata,ni2, ou1);   // **************
   rewind(ou1);
   rewind(in1);
   rewind(in2);

        printf("\n\n %s PSs %d\n",out1,n);
     fprintf(log,"\n %s PSs %d"  ,out1,n);

//-------------------------------------------------------------------

        printf("\n\n %s  PSs %d\n",argv[2],ni2);
   fprintf(log,"\n\n %s  PSs %d",  argv[2],ni2);

// Copy data to memory 
 
   free(indata);
   if (( indata = (psxy *) malloc(n * sizeof(psxy)) ) == NULL)
   { error("\nNot enough memory to allocate indata 2\n");  exit(1); }
   for(i=0;i<n;i++)
   {
     fscanf(ou1,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe);
     (indata+i)->la=la;
     (indata+i)->fi=fi;               
   }
//--------------------------------------------------------------------------

   printf("\n Select PSs ...\n");                                            
   n=selectp(dam, in2, indata,n, ou2);      // **************

      printf("\n\n %s PSs %d\n" ,out2,n);
   fprintf(log,"\n %s PSs %d\n\n",out2,n);

printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
        \n +                end   ps_data_select                +\
        \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(0);
      
}  // end data_select

 int dominant(int argc, char *argv[])
{
   int i, n1,n2,                      // number of data in input files
           nb=2,                      // starting number of data in cluster buffer, continiusly updated
             nc,                      // number of preselected clusters 
            nsc,                      // number of selected clusters 
            nhc,                      // number of hermit clusters             
            nps,                      // number of selected PSs in actual cluster
            ps1,                      // number of PSs from 1 input file
            ps2;                      // number of PSs from 2 input file 
                                  
   psxys *indata1,*indata2, *buffer;  // names of allocated memories
   char  *out="dominant.xyd",         // output file 
         *log="dominant.log";         // log output file

   FILE *in1,*in2,*ou,*lo;     

   float dam, la,fi,he,dhe,ve;

//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]);    
     
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
          \n +                      ps_dominant                      +\
          \n + clusters of ascending and descending PSs are selected +\
          \n +     and the dominant points (DSs) are estimated       +\
          \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc-Minarg<3)
 {
  printf("\n    usage:  ps_dominant asc_data.xys dsc_data.xys 100\n\
          \n            asc_data.xys   - (1st) ascending  data file\
          \n            dsc_data.xys   - (2nd) descending data file\
          \n            100            - (3rd) cluster separation (m)\n\
          \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
  exit(1);
 }

  if( (in1 = fopen(argv[1],"rt")) == NULL )
  { error("\n  ASC data file not found !\n"); exit(1); }
  if( (in2 = fopen(argv[2],"rt")) == NULL )
  { error("\n  DSC data file not found !\n"); exit(1); }
  if( (ou = fopen(out,"w+t")) == NULL )
  { error("\n  OUT data file not found !\n"); exit(1); }
  if( (lo = fopen(log,"w+t")) == NULL )
  { error("\n  LOG data file not found !\n"); exit(1); }

   fprintf(lo,"\n %s %s %s %s\n",argv[0],argv[1],argv[2],argv[3]);

       printf("\n  input: %s\n         %s\n",argv[1],argv[2]);
       printf("\n output: %s\n\n",out); 
   fprintf(lo,"\n  input: %s\n         %s\n",argv[1],argv[2]);
   fprintf(lo,"\n output: %s\n\n",out); 

//------------------------------------------------------
   sscanf(argv[3],"%f",&dam);
   printf("\n Appr. cluster size %5.1f (m)\n",dam); 
   fprintf(lo,"\n Appr. cluster size %5.1f (m)\n\n",dam); 
// -----------------------------------------------------

   printf("\n Copy data to memory ...\n");  
//   
   n1=0;
   while(fscanf(in1,"%e %e %e %e %e",&la,&fi,&ve,&he,&dhe)>0) n1++;
   rewind(in1); 
   
   if (( indata1 = (psxys *) malloc(n1 * sizeof(psxys)) ) == NULL)
   { error("\nNot enough memory to allocate indata 1\n");  exit(1); }
   for(i=0;i<n1;i++)
   {
     fscanf(in1,"%e %e %e %e %e",&la,&fi,&ve,&he,&dhe);
     (indata1+i)->ni = 1;     
     (indata1+i)->la = la;
     (indata1+i)->fi = fi;
     (indata1+i)->he = he+dhe;
     (indata1+i)->ve = ve;                    
   } fclose(in1);
   
   n2=0;
   while(fscanf(in2,"%e %e %e %e %e",&la,&fi,&ve,&he,&dhe)>0) n2++;
   rewind(in2); 
     
   if (( indata2 = (psxys *) malloc(n2 * sizeof(psxys)) ) == NULL)
   { error("\nNot enough memory to allocate indata 2\n");  exit(1); }
   for(i=0;i<n2;i++)
   {
     fscanf(in2,"%e %e %e %e %e",&la,&fi,&ve,&he,&dhe);
     (indata2+i)->ni = 2;        
     (indata2+i)->la = la;
     (indata2+i)->fi = fi;
     (indata2+i)->he = he+dhe;
     (indata2+i)->ve = ve;                                        
   } fclose(in2);

// ---------------------------------------------------------------

   if (( buffer = (psxys *) malloc(nb * sizeof(psxys)) ) == NULL)
   { error("\nNot enough memory to allocate buffer\n");  exit(1); }
//   
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  printf("\n selected clusters:\n"); 

       nps=nc=nhc=nsc=0;

   do{ nps = cluster(indata1,n1, indata2,n2, buffer,&nb, dam);   // ************

       ps1=ps2=0; 
       for(i=0;i<nps;i++) if((buffer+i)->ni==1) ps1++;
       else               if((buffer+i)->ni==2) ps2++;
       
       if((ps1*ps2)>0)
      { 
        estim_dominant(buffer, ps1, ps2, lo, ou);                // ************ 
        nsc++; 
      }
       else if ( (ps1+ps2) >0 ) nhc++;

       nc++;
       if((nc%2000)==0) printf("\n %6d ...",nc);  

     } while(nps > 0 );

        printf("\n %6d",nc-1);
     
        printf("\n\n hermit   clusters: %6d\n accepted clusters: %6d\n",nhc,nsc); 
        printf("\n Records of %s file:\n",out); 
        printf("\n longitude latitude  height asc_v dsc_v"); 
        printf("\n (     degree          m      mm/year )\n");  

      fprintf(lo,"\n hermit   clusters: %6d\n accepted clusters: %6d\n",nhc,nsc); 
      fprintf(lo,"\n Records of %s file:\n",out); 
      fprintf(lo,"\n longitude latitude  height asc_v dsc_v"); 
      fprintf(lo,"\n (     degree          m      mm/year )\n\n");  

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
        \n +                   end    ps_dominant                  +\
        \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(0);      
}  // end dominant   

 int integrate(int argc, char *argv[])
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
  { error("\nNot enough memory to allocate BUF\n");  exit(1); }

//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]);    
//  printf("%s\n",argv[3]);   
     
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
          \n +                       ds_integrate                          +\
          \n +        compute the east-west and up-down velocities         +\
          \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc-Minarg<3)
 {
  printf("\n usage:                                                      \n\
          \n    ds_integrate dominant.xyd asc_master.porb dsc_master.porb\n\
          \n              dominant.xyd  - (1st) dominant DSs data file   \
          \n           asc_master.porb  - (2nd) ASC polynomial orbit file\
          \n           dsc_master.porb  - (3rd) DSC polynomial orbit file\n\
          \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
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
  { error("\nNot enough memory to allocate POL1\n");  exit(1); }
    for(i=0;i<3;i++)
    for(j=0;j<dop1;j++)
    fscanf(ino1," %lf",(pol1+i*dop1+j));
    fclose(ino1);
// ---------------------------------------
  fscanf(ino2,"%d %lf %lf",&dop2,&ft2,&lt2);  // read orbit
  dop2++;
    if (( pol2 = (double *) malloc(dop2*3 * sizeof(double)) ) == NULL)
  { error("\nNot enough memory to allocate POL2\n");  exit(1); }
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
 
printf("\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
        \n +                     end    ds_integrate                     +\
        \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(0);
      
}  // end integrate

 int poly_orbit(int argc, char * argv[])
{
  int i,dop,               // deegre of polinomials
        ndp;               // number of orbit records
  torb *orb;               // tabular orbit data
  double *X, t,x,y,z;

  char *out, *buf, *head;
  char *log;              // log output file

  
  FILE *in,*ou, *lo;

  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
          \n +                     ps_poly_orbit                     +\
          \n +    tabular orbit data are converted to polynomials    +\
          \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc-Minarg<2)
 {
  printf("\n          usage:    ps_poly_orbit asc_master.res 4\
          \n                 or\
          \n                    ps_poly_orbit dsc_master.res 4\
          \n\n          asc_master.res or dsc_master.res - input files\
          \n          4                                - degree     \n\
          \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
  exit(1);
 }
    
//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]); 
  
  if (( out = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\nNot enough memory to allocate OUT\n"); exit(1); }
  if (( log = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\nNot enough memory to allocate LOG\n"); exit(1); }


  if (( buf = (char *) malloc(80 * sizeof(char)) )  == NULL)
  { error("\nNot enough memory to allocate BUF\n");  exit(1); } 
  if (( head = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\nNot enough memory to allocate HEAD\n"); exit(1); } 

  head="NUMBER_OF_DATAPOINTS:"; 
  sscanf(argv[2],"%d",&dop); 

//  sprintf(out,"%s%s",argv[1],"o"); 
      sprintf( out,"%s",argv[1] ); 
  change_ext ( out,"porb" );

  sprintf(log,"%s%s",out,".log"); 
      
  if( (in = fopen(argv[1],"rt")) == NULL )
  { error("\n  1. Data file not found !\n"); exit(1); }
  if( (ou = fopen(out,"w+t")) == NULL )
  { error("\n  3. Data file not found !\n"); exit(1); }
  if( (lo = fopen(log,"w+t")) == NULL )
  { error("\n  4. Data file not found !\n"); exit(1); }

   fprintf(lo,"\n %s %s %s\n",argv[0], argv[1],argv[2]); 
   fprintf(lo,"\n  input: %s",argv[1]);
   fprintf(lo,"\n output: %s",out); 
   fprintf(lo,"\n degree: %d\n",dop); 

   printf("\n  input: %s",argv[1]);
   printf("\n output: %s",out); 
   printf("\n degree: %d\n",dop); 

   while( fscanf(in,"%s",buf)>0 && strncmp(buf,head,21) !=0 ); 
   fscanf(in,"%d",&ndp);
   
   if (( orb = (torb *) malloc(ndp * sizeof(torb)) ) == NULL)
  { error("\nNot enough memory to allocate orb\n"); exit(1); } 
   if (( X = (double *) calloc((dop+1), sizeof(double)) ) == NULL)
  { error("\nNot enough memory to allocate X\n"); exit(1); } 
    
  for(i=0;i<ndp;i++)
{ 
  fscanf(in,"%lf %lf %lf %lf",&t,&x,&y,&z);
  (orb+i)->t=t;
  (orb+i)->x=x;
  (orb+i)->y=y;
  (orb+i)->z=z;  
} 
  fprintf(ou,"%3d\n",dop);
  fprintf(ou,"%13.5f\n", orb->t);
  fprintf(ou,"%13.5f\n",(orb+ndp-1)->t);    
//
      printf("\n fit of X coordinates:");
  fprintf(lo,"\n fit of X coordinates:");
  poly_fit ( ndp, dop+1,orb, X, 'x',lo );  // ***********
  for(i=0;i<(dop+1);i++) fprintf(ou," %23.15e",*(X+i));   
  fprintf(ou,"\n");  

       printf("\n\n fit of Y coordinates:");
  fprintf(lo,"\n\n fit of Y coordinates:");
  poly_fit ( ndp, dop+1,orb, X, 'y',lo );  // ***********
  for(i=0;i<(dop+1);i++) fprintf(ou," %23.15e",*(X+i));  
  fprintf(ou,"\n");
    
       printf("\n\n fit of Z coordinates:");
  fprintf(lo,"\n\n fit of Z coordinates:");
  poly_fit ( ndp, dop+1,orb, X, 'z',lo );  // ***********
  for(i=0;i<(dop+1);i++) fprintf(ou," %23.15e",*(X+i));  
  fprintf(ou,"\n\n");  

  fprintf(lo,"\n\n");

  fclose(in);
  fclose(ou);
  fclose(lo);

  printf("\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
          \n +             end          ps_poly_orbit                +\
          \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
     
}  // end poly_orbit

static int zero_select(int argc, char *argv[])
{
   int n=0,nz=0,nt=0;
   
   char *inp;    // input file
   char *out1;   // output file
   char *out2;   // output file

   char *buf;                       
   char *log="zero_select.log";  

   FILE *in, *ou1, *ou2, *lo;     

   float zch;
   float la,fi,he, ve,vu;
   
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                   ds_zero_select                   +\
            \n +  select integrated DSs with nearly zero velocity   +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]);    

    if(argc-Minarg<2)
   {
    printf("\n     usage:   ds_zero_select integrate.xyi 0.6 \n\
            \n            integrate.xyi  -  integrated data file\
            \n            0.6 (mm/year)  -  zero data criteria\n\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
    exit(1);
   }

  if (( inp = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\nNot enough memory to allocate INP\n");  exit(1); }
  if (( out1 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\nNot enough memory to allocate OUT\n");  exit(1); }
  if (( out2 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\nNot enough memory to allocate OUT\n");  exit(1); }
  if (( buf = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { error("\nNot enough memory to allocate buf\n");  exit(1); }
  
//   sprintf(out1,"%s%s",argv[1],".0ref");  // DSs smaller than criteria
//   sprintf(out2,"%s%s",argv[1],".0trg");  // DSs target

      sprintf(out1,"%s",argv[1]);  // DSs smaller than criteria
      sprintf(out2,"%s",argv[1]);  // DSs target
   change_ext(out1,"0ref");
   change_ext(out2,"0trg");

   sscanf(argv[2],"%f",&zch);

  if( (lo = fopen(log,"w+t")) == NULL )
  { error("\n LOG data file not found !\n"); exit(1); }

   fprintf(lo,"\n %s %s %s\n",argv[0],argv[1],argv[2]);
   
   printf("\n    input: %s"  ,argv[1]);
   printf("\n   output: %s\n           %s\n",out1,out2);      
   printf("\n   zero DSs < |%3.1f| mm/year\n",zch); 

   fprintf(lo,"\n    input: %s"  ,argv[1]);
   fprintf(lo,"\n   output: %s\n           %s\n",out1,out2);      
   fprintf(lo,"\n   zero DSs < |%3.1f| mm/year\n",zch); 

  if( (in = fopen(argv[1],"rt")) == NULL )
  { errorln("\n %s data file not found !",argv[1]); exit(1); }
  if( (ou1 = fopen(out1,"w+t")) == NULL )
  { error("\n OUT1 data file not found !\n"); exit(1); }
  if( (ou2 = fopen(out2,"w+t")) == NULL )
  { error("\n OUT2 data file not found !\n"); exit(1); }
    
//---------------------------------------------------------------
     
   while( fscanf(in,"%f %f %f %f %f",&la,&fi,&he,&ve,&vu) >0 )
  { 

      if( sqrt(ve*ve+vu*vu) <= zch )
     {
        fprintf(ou1,"%16.7e %15.7e %9.3f %7.3f %7.3f\n",la,fi,he,ve,vu);        
        nz++;

  
      }
      else 
     {
       fprintf(ou2,"%16.7e %15.7e %9.3f %7.3f %7.3f\n",la,fi,he,ve,vu);  
       nt++;
     }
      n++; if((n%1000)==0) printf("\n %6d ...",n); 
  } 
      printf("\n %6d\n",n); 
  
         printf("\n   zero dominant DSs %6d\n        target   DSs %6d\n",nz,nt);

      printf("\n Records of output files:\n"); 
      printf("\n longitude latitude  height  ew_v   up_v"); 
      printf("\n (     degree          m       mm/year )\n");  

     fprintf(lo,"\n   zero dominant DSs %6d\n        target   DSs %6d\n",nz,nt); 

     fprintf(lo,"\n Records of output files:\n"); 
     fprintf(lo,"\n longitude latitude  height  ew_v   up_v"); 
     fprintf(lo,"\n (     degree          m       mm/year )\n\n");  


//----------------------------------------------------------------------
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +              end    ds_zero_select                 +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    

return(0);

}  // end zero_select 

/*****************
 * Main function *
 *****************/

int main(int argc, char **argv)
{
    int ret;
    
    if (argc < 2) {
        error("At least one argument (module name) is required.\
               \nModules to choose from: data_select, dominant, poly_orbit,\
               \nintegrate, zero_select.\n");
        return(-1);
    }
    
    if (Str_Select("data_select"))
        return data_select(argc, argv);
    
    else if(Str_Select("dominant"))
        return dominant(argc, argv);
    
    else if(Str_Select("poly_orbit"))
        return poly_orbit(argc, argv);
    
    else if(Str_Select("integrate"))
        return integrate(argc, argv);

    else if(Str_Select("zero_select"))
        return zero_select(argc, argv);
    else {
        errorln("Unrecognized module: %s", argv[1]);
        error("Modules to choose from: data_select, dominant, poly_orbit, "
              "integrate, zero_select.\n");
        return(-1);
    }
}
