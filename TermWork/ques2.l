%{
    #include<stdio.h>
    #include<string.h>
%}
%%
"/*"[^*]*"*/" ;
"//"[^\n]*    ;
. {
    fprintf(yyout,"%s",yytext);
}
%%
int yywrap() { return 1; }
int main() {
    extern FILE *yyin, *yyout;
    yyin = fopen("input.c", "r");
    yyout = fopen("out.c", "w");
    yylex();
    fclose(yyin);
    fclose(yyout);
    return 0;
}
