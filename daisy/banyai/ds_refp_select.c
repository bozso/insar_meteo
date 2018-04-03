//
//  ds_refp_select
//
//  Select chosen integrated DSs points 
//  using modified workarea.aref as input file
//
//  DSs are separated into two output files:
//  DSs in the reference area: workarea.pref
//  and the other: workarea.ptrg (target)
//
//  The mean velocities in reference area
//  are subtracted from all DSs.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int ni; float lo; float la; float he; float ve;  float vu; } integrate;

int    identify ( int i, integrate *mem, int nref, integrate *ref);
void change_ext ( char *name, char *ext );

void  pause (int i);

// ------------------------------------------------------
 
 int main(int argc, char *argv[])
{
   int i, ni, nd=0, nref=0;  
   float minlo,maxlo, minla, maxla;
   float lo,la,he,ve,vu;
   float mve, dve, vs1=0.0, vs11=0.0;
   float mvu, dvu, vs2=0.0, vs22=0.0;

   integrate  *mem, *ref;
   char *buf,*out1, *out2;                   // output file
   char *logf="refp_select.log"; 

   FILE *in,*inr, *ou1, *ou2, *log;     

//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]);   
 
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
  printf("\n +                        ds_refp_select                     +");    
  printf("\n +       use selected DSs points of the reference area       +");
  printf("\n +     mean values are subtracted from all integrated DSs    +");    
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc<3)
 {
  printf("\n usage:                                                       \n");
  printf("\n        ds_refp_select integrate.xyi workarea.aref            \n");
  printf("\n        integrate.xyi   -  (1st) integrated data file         ");
  printf("\n        workarea.aref   -  (2nd) modified reference area DSs  \n");
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");    
  exit(1);
 }     
 
  if (( out1 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate INPP");  exit(1); }
  if (( out2 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate INPP");  exit(1); }
  if (( buf = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate BUF");  exit(1); }
  
      sprintf(out1,"%s",argv[2]);  // DSs smaller than criteria
      sprintf(out2,"%s",argv[2]);  // DSs target
   change_ext(out1,"pref");
   change_ext(out2,"ptrg");



  if( (in = fopen(argv[1],"rt")) == NULL )
  { printf("\n  %s data file not found ! ",argv[1]); exit(1); }
  if( (inr = fopen(argv[2],"rt")) == NULL )
  { printf("\n  %s data file not found ! ",argv[1]); exit(1); }

  if( (ou1 = fopen(out1,"w+t")) == NULL )
  { printf("\n  %s data file not found ! ", out1);   exit(1); }
  if( (ou2 = fopen(out2,"w+t")) == NULL )
  { printf("\n  %s data file not found ! ", out2);   exit(1); }
  if( (log = fopen(logf,"w+t")) == NULL )
  { printf("\n  %s data file not found ! ", logf);   exit(1); }


   fprintf(log,"\n");
   for(i=0;i<argc;i++) fprintf(log," %s",argv[i]); 

   printf("\n  input: %s",argv[1]);
   printf("\n  input: %s\n",argv[2]);
   printf("\n output:  %s",out1); 
   printf("\n output:  %s\n",out2); 

   fprintf(log,"\n\n  input: %s",argv[1]);
   fprintf(log,"\n  input: %s\n",argv[2]);
   fprintf(log,"\n output:  %s",out1); 
   fprintf(log,"\n output:  %s\n",out2); 

   if (( ref = (integrate *) malloc(1 * sizeof(integrate)) ) == NULL)
   { printf("\nNot enough memory to allocate indata 1");  exit(1); }

//----------------------------------------------------------------  
// copy ref data to memory

   nref=0;
   while(fscanf(inr,"%f %f %f %f %f",&lo,&la,&he,&ve,&vu)>0)
  { 
     nref++;
     ref = (integrate *) realloc(ref,nref * sizeof(integrate)); 
    (ref+nref-1)->ni=ni;
    (ref+nref-1)->lo=lo;
    (ref+nref-1)->la=la;
    (ref+nref-1)->he=he;
    (ref+nref-1)->ve=ve;
    (ref+nref-1)->vu=vu;              
  } // end while

   if (( mem = (integrate *) malloc(1 * sizeof(integrate)) ) == NULL)
   { printf("\nNot enough memory to allocate indata 1");  exit(1); }

//----------------------------------------------------------------
// copy all DSs data to memory
  
   while(fscanf(in,"%f %f %f %f %f",&lo,&la,&he,&ve,&vu)>0)
  { 
     nd++;
     mem = (integrate *) realloc(mem,nd * sizeof(integrate)); 
    (mem+nd-1)->ni=ni;
    (mem+nd-1)->lo=lo;
    (mem+nd-1)->la=la;
    (mem+nd-1)->he=he;
    (mem+nd-1)->ve=ve;
    (mem+nd-1)->vu=vu;              
  } // end while

//------------------------------------------------------------- 

   printf("\n\n integrated DSs reference points:\n");
   for(i=0;i<nd;i++)
  {
     if( identify(i,mem,nref,ref) == 1 )
    {
       vs1 +=(mem+i)->ve; vs11 += (mem+i)->ve * (mem+i)->ve; 
       vs2 +=(mem+i)->vu; vs22 += (mem+i)->vu * (mem+i)->vu;
    
       printf("\n %f %f  %7.2f %7.2f",(mem+i)->lo,(mem+i)->la,(mem+i)->ve,(mem+i)->vu);
    }                      
  } // end for 
  
    mve = vs1/(nref*1.0);  dve = sqrt( ( vs11-(vs1*vs1)/(nref*1.0) ) / ( (nref-1)*1.0 ) );
    mvu = vs2/(nref*1.0);  dvu = sqrt( ( vs22-(vs2*vs2)/(nref*1.0) ) / ( (nref-1)*1.0 ) ); 
                   
   printf("\n\n reference DSs points %6d\n",nref);
   printf("\n mean values          %7.2f %7.2f"  ,mve,mvu);
   printf("\n dispersions          %7.2f %7.2f\n",dve,dvu);

   fprintf(log,"\n\n reference DSs points %6d\n",nref);
   fprintf(log,"\n mean values %7.2f %7.2f"  ,mve,mvu);
   fprintf(log,"\n dispersions %7.2f %7.2f\n",dve,dvu);

          printf("\n target  area   DSs %6d",nd-nref);

      printf("\n\n Records of output files:\n"); 
      printf("\n longitude latitude  height  ew_v   up_v"); 
      printf("\n (     degree          m       mm/year )\n");  

     fprintf(log,"\n target  area   DSs %6d\n",nd-nref);

     fprintf(log,"\n Records of output files:\n"); 
     fprintf(log,"\n longitude latitude  height  ew_v   up_v"); 
     fprintf(log,"\n (     degree          m       mm/year )\n\n");  

//----------------------------------------------------------------------------
   for(i=0;i<nd;i++)
  {
     if( identify(i,mem,nref,ref) == 1 )
    {
          fprintf(ou1,"%16.7le %15.7le %9.3f %8.3f %8.3f\n",
          (mem+i)->lo,(mem+i)->la,(mem+i)->he,(mem+i)->ve-mve,(mem+i)->vu-mvu );
    }
     else fprintf(ou2,"%16.7e %15.7e %9.3f %8.3f %8.3f\n",
          (mem+i)->lo,(mem+i)->la,(mem+i)->he,(mem+i)->ve-mve,(mem+i)->vu-mvu );

  } // end for 
     
printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
printf("\n +                   end    ds_refp_select                   +");
printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(1);  
  
} // end main
//****************************************************************************

// ------------------------------------------------------------
 int identify ( int i, integrate *mem, int nref, integrate *ref)
{
  int j; 
  
  for(j=0;j<nref;j++)
  if ( ((mem+i)->lo == (ref+j)->lo) && ((mem+i)->la == (ref+j)->la) ) return(1);

  return(0);

} // end identify 

// --------------------------------------
 void change_ext ( char *name, char *ext )
{
// change the extent of name for ext

  int  i=0;
  while( *(name+i) != '.' && *(name+i) != '\0') i++;
  *(name+i)='\0';

  sprintf(name,"%s.%s",name,ext);

 } // end change_ext

// ---------------------------------------------------------------------
  void pause(int i)
{
 printf("\n\n Pause(%1d) => Ctrl c to exit - Enter to continue !\n",i);
 getchar();
}
//*****************************
