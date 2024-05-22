%{
    #include<stdio.h>
    #include<stdlib.h>
%}
%%
"#"[^\n]* ;
. {fprintf(yyout,"%s",yytext);}
%%
int yywrap(){
    return 1;
}
int main(){
    extern FILE *yyin,*yyout;
    yyin = fopen("C1_W1_Assignment.py","r");
    yyout = fopen("Assignment.py","w");
    yylex();
    fclose(yyin);
    fclose(yyout);
}