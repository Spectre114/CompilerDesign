%{
    #include<stdio.h>
    #include<stdlib.h>
%}
%%
"<"[^>]*">" {printf("%s\n",yytext);}
. ;
%%
int yywrap(){
    return 1;
}
int main(){
    extern FILE *yyin;
    yyin = fopen("htmlInput.html","r");
    yylex();
    fclose(yyin);
}