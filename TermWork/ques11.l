%{
    #include<stdio.h>
    #include<stdlib.h>
%}
%%
"<"[^>]*">"|" " {fprintf(yyout,"%s",yytext);}
. ;
%%
int yywrap(){
    return 1;
}
int main(){
    extern FILE *yyin,*yyout;
    yyin = fopen("htmlInput.html","r");
    yyout = fopen("htmlTags.txt","w");
    yylex();
    fclose(yyin);
    fclose(yyout);
}