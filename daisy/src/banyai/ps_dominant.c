//  ps_dominant
//
//  input files are asc_data.xys and dsc_data.xys
//  produced by ps_data_select 
//
//  clusters of ascending and descending PSs are selected 
//  and their dominamt points (DSs) are estimated.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define R 6372000      // radius of Earth
#define C 57.295779513 // 180/pi

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define WA  6378137.0            // WGS-84
#define WB  6356752.3142         // WGS-84
#define E2  (WA*WA-WB*WB)/WA/WA
#define distance(x,y,z) sqrt(y*y+x*x+z*z)

typedef struct {  int ni; float la; float fi; float he; float ve; } psxys;
typedef struct { double x; double y; double z;                    
	      	 double f; double l; double h; } station;      // [m,rad]

int  cluster(psxys *indata1, int n1, psxys *indata2, int n2, psxys *buffer, int *nb, float dam);

void estim_dominant(psxys *buffer, int ps1, int ps2, FILE *lg, FILE *ou);

void cart_ell  ( station *sta );
void ell_cart  ( station *sta );

void pause (int i);

// ------------------------------------------------------
 
 int main(int argc, char *argv[])
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
     
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
  printf("\n +                      ps_dominant                      +");    
  printf("\n + clusters of ascending and descending PSs are selected +");
  printf("\n +     and the dominant points (DSs) are estimated       +");    
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc<4)
 {
  printf("\n    usage:  ps_dominant asc_data.xys dsc_data.xys 100\n");
  printf("\n            asc_data.xys   - (1st) ascending  data file"); 
  printf("\n            dsc_data.xys   - (2nd) descending data file");
  printf("\n            100            - (3rd) cluster separation (m)\n");      
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
  exit(1);
 }

  if( (in1 = fopen(argv[1],"rt")) == NULL )
  { printf("\n  ASC data file not found ! "); exit(1); }
  if( (in2 = fopen(argv[2],"rt")) == NULL )
  { printf("\n  DSC data file not found ! "); exit(1); }
  if( (ou = fopen(out,"w+t")) == NULL )
  { printf("\n  OUT data file not found ! "); exit(1); }
  if( (lo = fopen(log,"w+t")) == NULL )
  { printf("\n  LOG data file not found ! "); exit(1); }

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
   { printf("\nNot enough memory to allocate indata 1");  exit(1); }
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
   { printf("\nNot enough memory to allocate indata 2");  exit(1); }
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
   { printf("\nNot enough memory to allocate buffer");  exit(1); }
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

printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
printf("\n +                   end    ps_dominant                  +");
printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(1);      
}  // end main   
// ********************************************************************

// --------------------------------------------------------------------
 void estim_dominant(psxys *buffer, int ps1,int ps2, FILE *lo, FILE *ou)
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
  int  cluster(psxys *indata1, int n1, psxys *indata2, int n2, 
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
  
// ---------------------------------------------------------------------
  void pause(int i)
{
 printf("\n\n Pause(%1d) => Ctrl c to exit - Enter to continue !\n",i);
 getchar();
} // end pause

// ---------------------------

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
 
}  // end of cart_ell

// --------------------------

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

// **********************************************
