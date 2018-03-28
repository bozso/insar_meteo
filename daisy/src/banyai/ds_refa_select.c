//
//  ds_refa_select
//
//  Select integrated DSs in the reference area
//
//  DSs are separated into two output files:
//  DSs ins the reference area
//  and the others.
//
//  The mean velocities in reference area
//  are subtracted from all DSs.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int ni; float lo; float la; float he; float ve;  float vu; } integrate;

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

   integrate  *mem;
   char *buf,*out1, *out2;                   // output file
   char *logf="refa_select.log"; 

   FILE *in, *ou1, *ou2, *log;     

//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]);   
 
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
  printf("\n +                      ds_refa_select                       +");    
  printf("\n +        select integrated DSs in the reference area        +");
  printf("\n +    mean values are subtracted from all integrated DSs     +");    
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc<7)
 {
  printf("\n usage:     \n");
  printf("\n   ds_refa_select integrate.xyi workarea 25.9 26.0 46.3 46.4 \n");
  printf("\n           integrate.xyi   -  (1st) integrated data file");
  printf("\n           workarea        -  (2nd) name of working area");
  printf("\n           25.9            -  (3rd) min. longitude"); 
  printf("\n           26.0            -  (4st) max. longitude");
  printf("\n           46.3            -  (5st) min. latitude");
  printf("\n           46.4            -  (6st) max. latitude\n");   
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
  exit(1);
 }     
 
  if (( out1 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate INPP");  exit(1); }
  if (( out2 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate INPP");  exit(1); }
  if (( buf = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate BUF");  exit(1); }
  
//    sprintf(out1,"%s%s",argv[2],".xyi.ref"); // reference area
//    sprintf(out2,"%s%s",argv[2],".xyi.trg"); // target area

      sprintf(out1,"%s",argv[2]);  // DSs smaller than criteria
      sprintf(out2,"%s",argv[2]);  // DSs target
   change_ext(out1,"aref");
   change_ext(out2,"atrg");



  if( (in = fopen(argv[1],"rt")) == NULL )
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
   printf("\n output:  %s",out1); 
   printf("\n output:  %s\n",out2); 

   fprintf(log,"\n\n  input: %s\n",argv[1]);
   fprintf(log,"\n output:  %s",out1); 
   fprintf(log,"\n output:  %s\n",out2); 


//   minlo=25.99;
//   maxlo=26.0; 
//   minla=46.35;     
//   maxla=46.4;  

     sscanf(argv[3],"%f",&minlo);
     sscanf(argv[4],"%f",&maxlo);
     sscanf(argv[5],"%f",&minla);
     sscanf(argv[6],"%f",&maxla);
     
     printf("\n reference area in degree\n");
     printf("\n min. longitude: %8.4f",minlo);
     printf("\n max. longitude: %8.4f\n",maxlo);
     printf("\n min.  latitude: %8.4f",minla);
     printf("\n max.  latitude: %8.4f",maxla);

     fprintf(log,"\n reference area in degree\n");
     fprintf(log,"\n min. longitude: %8.4f",minlo);
     fprintf(log,"\n max. longitude: %8.4f\n",maxlo);
     fprintf(log,"\n min.  latitude: %8.4f",minla);
     fprintf(log,"\n max.  latitude: %8.4f\n",maxla);


   if (( mem = (integrate *) malloc(1 * sizeof(integrate)) ) == NULL)
   { printf("\nNot enough memory to allocate indata 1");  exit(1); }

//    fscanf(in,"%d %s",&i,buf);
//    fprintf(ou,"%d %s\n",i,buf);
//    fscanf(in,"%d %s",&i,buf);
//    fprintf(ou,"%d %s\n",i,buf);
//----------------------------------------------------------------  
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
   printf("\n\n integrated DSs in reference area:\n");
   for(i=0;i<nd;i++)
  {
     if( ((mem+i)->lo > minlo) && ((mem+i)->lo < maxlo) &&  
         ((mem+i)->la > minla) && ((mem+i)->la < maxla) )
    {
       nref++; 
       vs1 +=(mem+i)->ve; vs11 += (mem+i)->ve * (mem+i)->ve; 
       vs2 +=(mem+i)->vu; vs22 += (mem+i)->vu * (mem+i)->vu;
    
       printf("\n %2d  %f %f  %7.2f %7.2f",nref,(mem+i)->lo,(mem+i)->la,(mem+i)->ve,(mem+i)->vu);
       fprintf(log,"\n %2d  %f %f  %7.2f %7.2f",nref,(mem+i)->lo,(mem+i)->la,(mem+i)->ve,(mem+i)->vu);

    }                      
  } // end for 
  
    mve = vs1/(nref*1.0);  dve = sqrt( ( vs11-(vs1*vs1)/(nref*1.0) ) / ( (nref-1)*1.0 ) );
    mvu = vs2/(nref*1.0);  dvu = sqrt( ( vs22-(vs2*vs2)/(nref*1.0) ) / ( (nref-1)*1.0 ) ); 
                   
   printf("\n\n reference area DSs     %6d\n",nref);
   printf("\n mean values              %7.2f %7.2f"  ,mve,mvu);
   printf("\n dispersions              %7.2f %7.2f\n",dve,dvu);

   fprintf(log,"\n\n reference area DSs     %6d\n",nref);
   fprintf(log,"\n mean values              %7.2f %7.2f"  ,mve,mvu);
   fprintf(log,"\n dispersions              %7.2f %7.2f\n",dve,dvu);

          printf("\n target  area   DSs     %6d",nd-nref);

      printf("\n\n Records of output files:\n"); 
      printf("\n longitude latitude  height  ew_v   up_v"); 
      printf("\n (     degree          m       mm/year )\n");  

     fprintf(log,"\n target  area   DSs     %6d\n",nd-nref);

     fprintf(log,"\n Records of output files:\n"); 
     fprintf(log,"\n longitude latitude  height  ew_v   up_v"); 
     fprintf(log,"\n (     degree          m       mm/year )\n\n");  

//----------------------------------------------------------------------------
   for(i=0;i<nd;i++)
  {
     if( ((mem+i)->lo > minlo) && ((mem+i)->lo < maxlo) &&  
         ((mem+i)->la > minla) && ((mem+i)->la < maxla) )
    {
          fprintf(ou1,"%16.7le %15.7le %9.3f %8.3f %8.3f\n",
          (mem+i)->lo,(mem+i)->la,(mem+i)->he,(mem+i)->ve-mve,(mem+i)->vu-mvu );
    }
     else fprintf(ou2,"%16.7e %15.7e %9.3f %8.3f %8.3f\n",
          (mem+i)->lo,(mem+i)->la,(mem+i)->he,(mem+i)->ve-mve,(mem+i)->vu-mvu );

  } // end for 
     
printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
printf("\n +                   end    ds_refa_select                   +");
printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(1);    
} // end main
//**************************************************

// -------------------------------------------------
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
