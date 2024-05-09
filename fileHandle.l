%{
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
%}
%%
[0-9]+ {
    fprintf(yyout,"Integer: %s\n",yytext);
}
[0-9]+"."[0-9]+ {
    fprintf(yyout,"Float: %s",yytext);
}
%%
int yywrap() {
    return 1;
}
int main(){
    extern FILE *yyin,*yyout;
    yyin=fopen("input.txt","r");
    yyout=fopen("output.txt","w");
    yylex();
    fclose(yyin);
    fclose(yyout);
    return 0;
}

