%{
    #include<stdio.h>
    #include<stdlib.h>
%}
%%
[0-9]+ {fprintf(yyout,"%s is an Integer\n",yytext);}
[0-9]*"."[0-9]+ {fprintf(yyout,"%s is a Float\n",yytext);}
"if"|"else"|"for"|"while"|"int"|"float" {fprintf(yyout,"%s is a Keyword\n",yytext);}
"+"|"-"|"*"|"/"|"=="|"="|"<"|">"|"++"|"--" {fprintf(yyout,"%s is an Operator\n",yytext);}
","|";" {fprintf(yyout,"%s is A Seperator\n",yytext);}
[a-zA-Z][a-zA-Z0-9]* {fprintf(yyout,"%s is an Identifier\n",yytext);}
[ ] {printf("");} 
. ;
%%
int yywrap(){return 1;}
int main(){
    extern FILE *yyin,*yyout;
    yyin = fopen("token.c","r");
    yyout = fopen("tokens.txt","w");
    yylex();
    fclose(yyin);
    fclose(yyout);
}
