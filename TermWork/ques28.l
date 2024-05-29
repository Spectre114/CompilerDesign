%{
    #include<stdio.h>
%}
%s A B C D E F G
%%
<INITIAL>a BEGIN A;
<INITIAL>b BEGIN INITIAL;
<INITIAL>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<A>a BEGIN B;
<A>b BEGIN C;
<A>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<B>a BEGIN D;
<B>b BEGIN E;
<B>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<C>a BEGIN F;
<C>b BEGIN G;
<C>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<D>a BEGIN D;
<D>b BEGIN E;
<D>\n BEGIN INITIAL; {printf("Accepted\n");}
<E>a BEGIN F;
<E>b BEGIN G;
<E>\n BEGIN INITIAL; {printf("Accepted\n");}
<F>a BEGIN B;
<F>b BEGIN C;
<F>\n BEGIN INITIAL; {printf("Accepted\n");}
<G>a BEGIN A;
<G>b BEGIN INITIAL;
<G>\n BEGIN INITIAL; {printf("Accepted\n");}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the Input\n");
    yylex();
    return 0;
}