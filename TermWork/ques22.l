%{
    #include<stdio.h>
%}
%s A B C D E F G
%%
<INITIAL>0 BEGIN A;
<INITIAL>1 BEGIN D;
<INITIAL>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<A>0 BEGIN D;
<A>1 BEGIN B;
<A>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<B>1 BEGIN C;
<B>0 BEGIN D;
<B>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<C>0 BEGIN E;
<C>1 BEGIN C;
<C>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<D>0 BEGIN D;
<D>1 BEGIN D;
<D>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<E>0 BEGIN F;
<E>1 BEGIN G;
<E>\n BEGIN INITIAL; {printf("Accepted\n");}
<F>0 BEGIN F;
<F>1 BEGIN G;
<F>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<G>0 BEGIN F;
<G>1 BEGIN C;
<G>\n BEGIN INITIAL; {printf("Not Accepted\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}
