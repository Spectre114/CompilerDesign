%{
    #include<stdio.h>
%}
%s A B C D E
%%
<INITIAL>a BEGIN A;
<INITIAL>b BEGIN D;
<INITIAL>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<A>a BEGIN D;
<A>b BEGIN B;
<A>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<B>b BEGIN B;
<B>a BEGIN C;
<B>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<C>a BEGIN E;
<C>b BEGIN B;
<C>\n BEGIN INITIAL; {printf("Accepted\n");}
<D>a BEGIN D;
<D>b BEGIN D;
<D>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<E>a BEGIN E;
<E>b BEGIN B;
<E>\n BEGIN INITIAL; {printf("Not Accepted\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}
