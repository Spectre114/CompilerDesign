%{
    #include<stdio.h>
%}
%s A B C D E
%%
<INITIAL>0 BEGIN A;
<INITIAL>1 BEGIN D;
<INITIAL>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<A>0 BEGIN D;
<A>1 BEGIN B;
<A>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<B>1 BEGIN B;
<B>0 BEGIN C;
<B>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<C>0 BEGIN E;
<C>1 BEGIN B;
<C>\n BEGIN INITIAL; {printf("Accepted\n");}
<D>0 BEGIN D;
<D>1 BEGIN D;
<D>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<E>0 BEGIN E;
<E>1 BEGIN B;
<E>\n BEGIN INITIAL; {printf("Not Accepted\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}
